#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <stdarg.h>
#include "svm.h"

typedef float Qfloat;
typedef signed char schar;
#ifndef min
template <class T> inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class T> inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
template <class S, class T> inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
inline double powi(double base, int times)
{
        double tmp = base, ret = 1.0;

        for(int t=times; t>0; t/=2)
	{
                if(t%2==1) ret*=tmp;
                tmp = tmp * tmp;
        }
        return ret;
}
#define INF HUGE_VAL
#define TAU 1e-12
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#if 1
void info(char *fmt,...)
{
	va_list ap;
	va_start(ap,fmt);
	vprintf(fmt,ap);
	va_end(ap);
}
void info_flush()
{
	fflush(stdout);
}
#else
void info(char *fmt,...) {}
void info_flush() {}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel Cache
//
// l is the number of total data items
// size is the cache size limit in bytes
class Cache
{
public:
	Cache(int l,int size);
	~Cache();

	// request data [0,len)
	// return some position p where [p,len) need to be filled
	// (p >= len if nothing needs to be filled)
	int get_data(const int index, Qfloat **data, int len);
	void swap_index(int i, int j);	// future_option
private:
	int l;
	int size;
	struct head_t
	{
		head_t *prev, *next;	// a cicular list
		Qfloat *data;
		int len;		// data[0,len) is cached in this entry
	};

	head_t *head;
	head_t lru_head;
	void lru_delete(head_t *h);
	void lru_insert(head_t *h);
};

Cache::Cache(int l_,int size_):l(l_),size(size_)
{
	head = (head_t *)calloc(l,sizeof(head_t));	// initialized to 0
	size /= sizeof(Qfloat);			//translate size into the number of Qfloat
	size -= l * sizeof(head_t) / sizeof(Qfloat);	//deduct the size of the head for l data
	size = max(size, 2*l);	// cache must be large enough for two columns
	lru_head.next = lru_head.prev = &lru_head;
}

Cache::~Cache()
{
	for(head_t *h = lru_head.next; h != &lru_head; h=h->next)
		free(h->data);
	free(head);
}

void Cache::lru_delete(head_t *h)
{
	// delete from current location
	h->prev->next = h->next;
	h->next->prev = h->prev;
}

void Cache::lru_insert(head_t *h)
{
	// insert to last position
	h->next = &lru_head;
	h->prev = lru_head.prev;
	h->prev->next = h;
	h->next->prev = h;
}

// return some position p where [p,len) need to be filled
// (p >= len if nothing needs to be filled)
int Cache::get_data(const int index, Qfloat **data, int len)
{
	head_t *h = &head[index];
	
	if(h->len) lru_delete(h);	
	int more = len - h->len;

	//more > 0 when part of this row is not in memory
	//in this case, we need to free some old space and copy 
	//the complete row into memory
	if(more > 0)
	{
		// free old space from the beginning
		while(size < more)
		{
			head_t *old = lru_head.next;
			lru_delete(old);
			free(old->data);
			size += old->len;
			old->data = 0;
			old->len = 0;
		}

		//allocate new space
		h->data = (Qfloat *)realloc(h->data,sizeof(Qfloat)*len);
		size -= more;
		swap(h->len,len);
	}

	//put the new row into the last position, which indicates
	//it has been used very recently
	lru_insert(h);
	*data = h->data;
	return len;
}

void Cache::swap_index(int i, int j)
{
	if(i==j) return;

	if(head[i].len) lru_delete(&head[i]);
	if(head[j].len) lru_delete(&head[j]);
	swap(head[i].data,head[j].data);
	swap(head[i].len,head[j].len);
	if(head[i].len) lru_insert(&head[i]);
	if(head[j].len) lru_insert(&head[j]);

	if(i>j) swap(i,j);
	for(head_t *h = lru_head.next; h!=&lru_head; h=h->next)
	{
		if(h->len > i)
		{
			if(h->len > j)
				swap(h->data[i],h->data[j]);
			else
			{
				// give up
				lru_delete(h);
				free(h->data);
				size += h->len;
				h->data = 0;
				h->len = 0;
			}
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel evaluation
//
// the static method k_function is for doing single kernel evaluation
// the constructor of Kernel prepares to calculate the l*l kernel matrix
// the member function get_Q is for getting one column from the Q Matrix
//
class QMatrix {
public:
	virtual Qfloat *get_Q(int column, int len) const = 0;
	virtual Qfloat *get_QD() const = 0;
	virtual void swap_index(int i, int j) const = 0;
	virtual ~QMatrix() {}
};

class Kernel: public QMatrix {
public:
	Kernel(int l, svm_node * const * x, const svm_parameter& param);
	Kernel(int l, int k, const double* aux_fx);
	virtual ~Kernel();
	void init_kernel();

	static double k_function(const svm_node *x, const svm_node *y,
				 const svm_parameter& param);
	virtual Qfloat *get_Q(int column, int len) const = 0;
	virtual Qfloat *get_QD() const = 0;
	virtual void swap_index(int i, int j) const	// no so const...
	{
		if(x) swap(x[i],x[j]);
		if(x_square) swap(x_square[i],x_square[j]);
		if(aux_fx)
            for(int t=0; t<k; t++)
				swap(aux_fx[t*l + i], aux_fx[t*l + j]);
	}
protected:

	double (Kernel::*kernel_function)(int i, int j) const;

private:
	const svm_node **x;	//data x[i]
	double *x_square;	//<x[i], x[i]>
	double *aux_fx;		// f_k(x_i)
	int l;				//number of data 
	int k;				//num of auxiliary model

	// svm_parameter
	const int kernel_type;
	const int degree;
	const double gamma;
	const double coef0;

	static double dot(const svm_node *px, const svm_node *py);
	double kernel_linear(int i, int j) const
	{
		return dot(x[i],x[j]);
	}
	double kernel_poly(int i, int j) const
	{
		return powi(gamma*dot(x[i],x[j])+coef0,degree);
	}
	double kernel_rbf(int i, int j) const
	{
		return exp(-gamma*(x_square[i]+x_square[j]-2*dot(x[i],x[j])));
	}
	double kernel_sigmoid(int i, int j) const
	{
		return tanh(gamma*dot(x[i],x[j])+coef0);
	}
	double kernel_precomputed(int i, int j) const
	{
		return x[i][(int)(x[j][0].value)].value;
	}
	double kernel_auxiliary(int i, int j) const
	{
		double ret = 0;
		for(int t=0; t<k; t++)
			ret += (aux_fx[t*l + i] * aux_fx[t*l + j]);
		return ret;
	}
};

Kernel::Kernel(int l, svm_node * const * x_, const svm_parameter& param)
:kernel_type(param.kernel_type), degree(param.degree),gamma(param.gamma), coef0(param.coef0), x(0), x_square(0), aux_fx(0)
{
	this->l = l;
	clone(x,x_,l);

	if(kernel_type == RBF)
	{
		x_square = new double[l];
		for(int i=0;i<l;i++)
			x_square[i] = dot(x[i],x[i]);
	}
	
	init_kernel();
}

Kernel::Kernel(int l, int k, const double* aux_fx_) 
: kernel_type(AUXILIARY), degree(0),gamma(0), coef0(0), x(0), x_square(0), aux_fx(0)
{
	this->l = l;
	this->k = k;	
	clone(aux_fx,aux_fx_,l*k);
	
	init_kernel();
}


Kernel::~Kernel()
{
	if(x)		delete[] x;
	if(aux_fx)	delete[] aux_fx;
	if(x_square)	delete[] x_square;
}

void Kernel::init_kernel()
{
	switch(kernel_type)
	{
		case LINEAR:
			kernel_function = &Kernel::kernel_linear;
			break;
		case POLY:
			kernel_function = &Kernel::kernel_poly;
			break;
		case RBF:
			kernel_function = &Kernel::kernel_rbf;
			break;
		case SIGMOID:
			kernel_function = &Kernel::kernel_sigmoid;
			break;
		case PRECOMPUTED:
			kernel_function = &Kernel::kernel_precomputed;
			break;
		case AUXILIARY:
			kernel_function = &Kernel::kernel_auxiliary;
			break;
	}
}

double Kernel::dot(const svm_node *px, const svm_node *py)
{
	double sum = 0;
	while(px->index != -1 && py->index != -1)
	{
		if(px->index == py->index)
		{
			sum += px->value * py->value;
			++px;
			++py;
		}
		else
		{
			if(px->index > py->index)
				++py;
			else
				++px;
		}			
	}
	return sum;
}

double Kernel::k_function(const svm_node *x, const svm_node *y,const svm_parameter& param)
{
	switch(param.kernel_type)
	{
		case LINEAR:
			return dot(x,y);
		case POLY:
			return powi(param.gamma*dot(x,y)+param.coef0,param.degree);
		case RBF:
		{
			double sum = 0;
			while(x->index != -1 && y->index !=-1)
			{
				if(x->index == y->index)
				{
					double d = x->value - y->value;
					sum += d*d;
					++x;
					++y;
				}
				else
				{
					if(x->index > y->index)
					{	
						sum += y->value * y->value;
						++y;
					}
					else
					{
						sum += x->value * x->value;
						++x;
					}
				}
			}

			while(x->index != -1)
			{
				sum += x->value * x->value;
				++x;
			}

			while(y->index != -1)
			{
				sum += y->value * y->value;
				++y;
			}
			
			return exp(-param.gamma*sum);
		}
		case SIGMOID:
			return tanh(param.gamma*dot(x,y)+param.coef0);
		case PRECOMPUTED:  //x: test (validation), y: SV
			return x[(int)(y->value)].value;		
		default:
			return 0;	/* Unreachable */
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generalized SMO+SVMlight algorithm
// Solves:
//
//	min 0.5(\alpha^T Q \alpha) + b^T \alpha
//
//		y^T \alpha = \delta
//		y_i = +1 or -1
//		0 <= alpha_i <= Cp for y_i = 1
//		0 <= alpha_i <= Cn for y_i = -1
//
// Given:
//
//	Q, b, y, Cp, Cn, and an initial feasible point \alpha
//	l is the size of vectors and matrices
//	eps is the stopping criterion
//
// solution will be put in \alpha, objective value will be put in obj
//
///////////////////////////////////////////////////////////////
//The following is the code for adpative SVM
///////////////////////////////////////////////////////////////

//
// Q matrices for various formulations (Q_ij = K_ij * y_i * y_j)
//
class SVC_Q: public Kernel
{ 
public:
	//construct
	SVC_Q(const svm_problem& prob, const svm_parameter& param, const schar *y_)
	:Kernel(prob.l, prob.x, param)
	{
		clone(y,y_,prob.l);
		cache = new Cache(prob.l,(int)(param.cache_size*(1<<20)));
		QD = new Qfloat[prob.l];
		for(int i=0;i<prob.l;i++)
			QD[i]= (Qfloat)(this->*kernel_function)(i,i);
	}

	SVC_Q(const svm_problem& prob, const svm_parameter& param, const schar *y_, const double* aux_fx)
	:Kernel(prob.l, param.num_aux, aux_fx)
	{
		clone(y,y_,prob.l);
		cache = new Cache(prob.l,(int)(param.cache_size*(1<<20)));
		QD = new Qfloat[prob.l];
		for(int i=0;i<prob.l;i++)
			QD[i]= (Qfloat)(this->*kernel_function)(i,i);
	}
	
	//get a line of Q matrix with specified length
	Qfloat *get_Q(int i, int len) const
	{
		Qfloat *data;
		int start;
		
		//when start < len, then [start, len) part of the elements in Q (K)
		//need to be recomputed from the kernel function
		if((start = cache->get_data(i,&data,len)) < len)
		{
			for(int j=start;j<len;j++)
				data[j] = (Qfloat)(y[i]*y[j]*(this->*kernel_function)(i,j));
		}
		return data;
	}

	Qfloat *get_QD() const
	{
		return QD;
	}

	void swap_index(int i, int j) const
	{
		cache->swap_index(i,j);
		Kernel::swap_index(i,j);
		swap(y[i],y[j]);
		swap(QD[i],QD[j]);
	}

	~SVC_Q()
	{
		delete[] y;
		delete cache;
		delete[] QD;
	}
private:
	schar *y;
	Cache *cache;
	Qfloat *QD;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// construct and solve various formulations
//

// decision_function
struct decision_function
{
	double *alpha;
	double rho;	
};

//compute the output of aux models on the training data (as a matrix)
void aux_model_predictions(const svm_problem *prob, const svm_parameter *param, struct svm_model** aux_models, double* F)
{
	for(int i = 0; i<prob->l; i++)
	{
		for(int k=0; k < param->num_aux; k++)
		{
			double score;
			svm_predict(aux_models[k], prob->x[i], &score);
			F[k*prob->l + i] = score;
		}
	}
}

//compute fa(x_i)y_i and fa(x_i) can be a weighted ensemble of multiple aux models
void aux_model_predictions_ensemble(const svm_problem *prob, const svm_parameter *param, struct svm_model** aux_models, double* aux_fx)
{
	for(int i = 0; i<prob->l; i++)
	{
		aux_fx[i] = 1;
		for(int k=0; k < param->num_aux; k++)
		{
			double score;
			svm_predict(aux_models[k], prob->x[i], &score);
			aux_fx[i] -= (score * param->aux_weights[k] * prob->y[i]);
		}
	}
}


int svm_get_svm_type(const svm_model *model)
{
	return model->param.svm_type;
}

int svm_get_nr_class(const svm_model *model)
{
	return model->nr_class;
}

void svm_get_labels(const svm_model *model, int* label)
{
	if (model->label != NULL)
		for(int i=0;i<model->nr_class;i++)
			label[i] = model->label[i];
}

double svm_get_svr_probability(const svm_model *model)
{
	info("Model doesn't contain information for SVR probability inference\n");
	return 0;
}

void svm_predict_values(const svm_model *model, const svm_node *x, double* dec_values)
{
	int i;
	int nr_class = model->nr_class;
	int l = model->l;
	
	double *kvalue = Malloc(double,l);
	for(i=0;i<l;i++)
		kvalue[i] = Kernel::k_function(x,model->SV[i],model->param);

	int *start = Malloc(int,nr_class);
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+model->nSV[i-1];

	int p=0;
	int pos=0;
	for(i=0;i<nr_class;i++)
		for(int j=i+1;j<nr_class;j++)
		{
			double sum = 0;
			int si = start[i];
			int sj = start[j];
			int ci = model->nSV[i];
			int cj = model->nSV[j];
			
			int k;
			double *coef1 = model->sv_coef[j-1];
			double *coef2 = model->sv_coef[i];
			for(k=0;k<ci;k++)
				sum += coef1[si+k] * kvalue[si+k];
			for(k=0;k<cj;k++)
				sum += coef2[sj+k] * kvalue[sj+k];
			sum -= model->rho[p++];
			dec_values[pos++] = sum;
		}

	free(kvalue);
	free(start);	
}

//Jun Yang: We add an argument of this function to hold the decision values 
double svm_predict(const svm_model *model, const svm_node *x, double *dec_value)
{
	int i;
	int nr_class = model->nr_class;
	double *dec_values = Malloc(double, nr_class*(nr_class-1)/2);		
	svm_predict_values(model, x, dec_values);
	(*dec_value) = dec_values[0];

	int *vote = Malloc(int,nr_class);
	for(i=0;i<nr_class;i++)
		vote[i] = 0;
	int pos=0;
	for(i=0;i<nr_class;i++)
		for(int j=i+1;j<nr_class;j++)
		{
			if(dec_values[pos++] > 0)
				++vote[i];
			else
				++vote[j];
		}

	int vote_max_idx = 0;
	for(i=1;i<nr_class;i++)
		if(vote[i] > vote[vote_max_idx])
			vote_max_idx = i;
	free(vote);
	free(dec_values);
	return model->label[vote_max_idx];
}

const char *svm_type_table[] =
{
	"c_svc","nu_svc","one_class","epsilon_svr","nu_svr",NULL
};

const char *kernel_type_table[]=
{
	"linear","polynomial","rbf","sigmoid","precomputed",NULL
};

int adapt_svm_save_model(const char *model_file_name, const svm_model *model, svm_model** aux_models)
{

	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	const svm_parameter& param = model->param;

	fprintf(fp,"svm_type %s\n", svm_type_table[param.svm_type]);
	fprintf(fp,"kernel_type %s\n", kernel_type_table[param.kernel_type]);

	if(param.kernel_type == POLY)
		fprintf(fp,"degree %d\n", param.degree);

	if(param.kernel_type == POLY || param.kernel_type == RBF || param.kernel_type == SIGMOID)
		fprintf(fp,"gamma %g\n", param.gamma);

	if(param.kernel_type == POLY || param.kernel_type == SIGMOID)
		fprintf(fp,"coef0 %g\n", param.coef0);

	int nr_class = model->nr_class;
	
	int l = model->l;
	for(int k=0; k<param.num_aux; k++)	l += aux_models[k]->l; 

	fprintf(fp, "nr_class %d\n", nr_class);
	fprintf(fp, "total_sv %d\n",l);

	{
		fprintf(fp, "rho");
		for(int i=0;i<nr_class*(nr_class-1)/2;i++)
			fprintf(fp," %g",model->rho[i]);
		fprintf(fp, "\n");
	}
	
	if(model->label)
	{
		fprintf(fp, "label");
		for(int i=0;i<nr_class;i++)
			fprintf(fp," %d",model->label[i]);
		fprintf(fp, "\n");
	}

	if(model->probA) // regression has probA only
	{
		fprintf(fp, "probA");
		for(int i=0;i<nr_class*(nr_class-1)/2;i++)
			fprintf(fp," %g",model->probA[i]);
		fprintf(fp, "\n");
	}
	if(model->probB)
	{
		fprintf(fp, "probB");
		for(int i=0;i<nr_class*(nr_class-1)/2;i++)
			fprintf(fp," %g",model->probB[i]);
		fprintf(fp, "\n");
	}

	if(model->nSV)
	{
		fprintf(fp, "nr_sv");
		for(int i=0;i<nr_class;i++)
		{
			int nSV = model->nSV[i];
			for(int k=0; k<param.num_aux; k++)	
				nSV += aux_models[k]->nSV[i];
            fprintf(fp," %d",nSV);
		}	
		fprintf(fp, "\n");
	}
	fprintf(fp, "SV\n");
	
	for(int i=0;i<model->l;i++)
	{
		for(int j=0;j<nr_class-1;j++)
			fprintf(fp, "%.16g ",model->sv_coef[j][i]);

		const svm_node *p = model->SV[i];		

		if(param.kernel_type == PRECOMPUTED)
			fprintf(fp,"0:%d ",(int)(p->value));
		else
			while(p->index != -1)
			{
				fprintf(fp,"%d:%.8g ",p->index,p->value);
				p++;
			}
		fprintf(fp, "\n");
	}

	for(k=0; k<param.num_aux; k++)
	{
		for(i=0;i<aux_models[k]->l;i++)
		{
			for(int j=0;j<nr_class-1;j++)
				fprintf(fp, "%.16g ",aux_models[k]->sv_coef[j][i] * param.aux_weights[k]);

			const svm_node *p = aux_models[k]->SV[i];		

			if(param.kernel_type == PRECOMPUTED)
				fprintf(fp,"0:%d ",(int)(p->value));
			else
				while(p->index != -1)
				{
					fprintf(fp,"%d:%.8g ",p->index,p->value);
					p++;
				}
			fprintf(fp, "\n");
		}
	}

	fclose(fp);

	return 0;
}

svm_model *svm_load_model(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"rb");
	if(fp==NULL) return NULL;
	
	// read parameters
	svm_model *model = Malloc(svm_model,1);
	svm_parameter& param = model->param;
	model->rho = NULL;
	model->probA = NULL;
	model->probB = NULL;
	model->label = NULL;
	model->nSV = NULL;

	char cmd[81];
	while(1)
	{
		fscanf(fp,"%80s",cmd);

		if(strcmp(cmd,"svm_type")==0)
		{
			fscanf(fp,"%80s",cmd);
			int i;
			for(i=0;svm_type_table[i];i++)
			{
				if(strcmp(svm_type_table[i],cmd)==0)
				{
					param.svm_type=i;
					break;
				}
			}
			if(svm_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown svm type.\n");
				free(model->rho);
				free(model->label);
				free(model->nSV);
				free(model);
				return NULL;
			}
		}
		else if(strcmp(cmd,"kernel_type")==0)
		{		
			fscanf(fp,"%80s",cmd);
			int i;
			for(i=0;kernel_type_table[i];i++)
			{
				if(strcmp(kernel_type_table[i],cmd)==0)
				{
					param.kernel_type=i;
					break;
				}
			}
			if(kernel_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown kernel function.\n");
				free(model->rho);
				free(model->label);
				free(model->nSV);
				free(model);
				return NULL;
			}
		}
		else if(strcmp(cmd,"degree")==0)
			fscanf(fp,"%d",&param.degree);
		else if(strcmp(cmd,"gamma")==0)
			fscanf(fp,"%lf",&param.gamma);
		else if(strcmp(cmd,"coef0")==0)
			fscanf(fp,"%lf",&param.coef0);
		else if(strcmp(cmd,"nr_class")==0)
			fscanf(fp,"%d",&model->nr_class);
		else if(strcmp(cmd,"total_sv")==0)
			fscanf(fp,"%d",&model->l);
		else if(strcmp(cmd,"rho")==0)
		{
			int n = model->nr_class * (model->nr_class-1)/2;
			model->rho = Malloc(double,n);
			for(int i=0;i<n;i++)
				fscanf(fp,"%lf",&model->rho[i]);
		}
		else if(strcmp(cmd,"label")==0)
		{
			int n = model->nr_class;
			model->label = Malloc(int,n);
			for(int i=0;i<n;i++)
				fscanf(fp,"%d",&model->label[i]);
		}
		else if(strcmp(cmd,"probA")==0)
		{
			int n = model->nr_class * (model->nr_class-1)/2;
			model->probA = Malloc(double,n);
			for(int i=0;i<n;i++)
				fscanf(fp,"%lf",&model->probA[i]);
		}
		else if(strcmp(cmd,"probB")==0)
		{
			int n = model->nr_class * (model->nr_class-1)/2;
			model->probB = Malloc(double,n);
			for(int i=0;i<n;i++)
				fscanf(fp,"%lf",&model->probB[i]);
		}
		else if(strcmp(cmd,"nr_sv")==0)
		{
			int n = model->nr_class;
			model->nSV = Malloc(int,n);
			for(int i=0;i<n;i++)
				fscanf(fp,"%d",&model->nSV[i]);
		}
		else if(strcmp(cmd,"SV")==0)
		{
			while(1)
			{
				int c = getc(fp);
				if(c==EOF || c=='\n') break;	
			}
			break;
		}
		else
		{
			fprintf(stderr,"unknown text in model file\n");
			free(model->rho);
			free(model->label);
			free(model->nSV);
			free(model);
			return NULL;
		}
	}

	// read sv_coef and SV
	int elements = 0;
	long pos = ftell(fp);
	while(1)
	{
		int c = fgetc(fp);
		switch(c)
		{
			case '\n':
				// count the '-1' element
			case ':':
				++elements;
				break;
			case EOF:
				goto out;
			default:
				;
		}
	}
out:
	fseek(fp,pos,SEEK_SET);

	int m = model->nr_class - 1;
	int l = model->l;
	model->sv_coef = Malloc(double *,m);
	int i;
	for(i=0;i<m;i++)
		model->sv_coef[i] = Malloc(double,l);
	model->SV = Malloc(svm_node*,l);
	svm_node *x_space=NULL;
	if(l>0) x_space = Malloc(svm_node,elements);

	int j=0;
	for(i=0;i<l;i++)
	{
		model->SV[i] = &x_space[j];
		for(int k=0;k<m;k++)
			fscanf(fp,"%lf",&model->sv_coef[k][i]);
		while(1)
		{
			int c;
			do {
				c = getc(fp);
				if(c=='\n') goto out2;
			} while(isspace(c));
			ungetc(c,fp);
			fscanf(fp,"%d:%lf",&(x_space[j].index),&(x_space[j].value));
			++j;
		}	
out2:
		x_space[j++].index = -1;
	}

	fclose(fp);

	model->free_sv = 1;	// XXX
	return model;
}

void svm_destroy_model(svm_model* model)
{
	if(model->free_sv && model->l > 0)
		free((void *)(model->SV[0]));
	for(int i=0;i<model->nr_class-1;i++)
		free(model->sv_coef[i]);
	free(model->SV);
	free(model->sv_coef);
	free(model->rho);
	free(model->label);
	free(model->probA);
	free(model->probB);
	free(model->nSV);
	free(model);
}

void svm_destroy_param(svm_parameter* param)
{
	free(param->weight_label);
	free(param->weight);
}

const char *svm_check_parameter(const svm_problem *prob, const svm_parameter *param)
{
	// svm_type

	int svm_type = param->svm_type;
	if(svm_type != C_SVC)
		return "only support svm classifier (C-SVC) in this package";
	
	// kernel_type, degree	
	int kernel_type = param->kernel_type;
	if(kernel_type != LINEAR &&
	   kernel_type != POLY &&
	   kernel_type != RBF &&
	   kernel_type != SIGMOID &&
	   kernel_type != PRECOMPUTED)
		return "unknown kernel type";

	if(param->degree < 0)
		return "degree of polynomial kernel < 0";

	// cache_size,eps,C,nu,p,shrinking
	if(param->cache_size <= 0)
		return "cache_size <= 0";

	if(param->eps <= 0)
		return "eps <= 0";

	if(svm_type == C_SVC)
		if(param->C <= 0)
			return "C <= 0";

	if(param->shrinking != 0 && param->shrinking != 1)
		return "shrinking != 0 and shrinking != 1";

	if(param->probability != 0)
		return "probability != 0";

	return NULL;
}

int svm_check_probability_model(const svm_model *model)
{
	return (model->param.svm_type == C_SVC && model->probA!=NULL && model->probB!=NULL);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//QP solver for adaptive SVM

class AdaptSolver {
public:
	AdaptSolver(bool w_, int l_, int na_) : is_auto_weight(w_), l(l_), na(na_) {};
	virtual ~AdaptSolver() {};

	struct AdaptSolutionInfo {
		double obj;
		double rho;
		double upper_bound_p;
		double upper_bound_n;
		double r;	// for Solver_NU
	};

	void Solve(const QMatrix& Q, const double *aux_fx_, const schar *y_, double *alpha_, double Cp, double Cn, double eps,
		   AdaptSolutionInfo* si, int shrinking);

	void Solve_w(const QMatrix& Q, const QMatrix& F, const schar *y_,double *alpha_, double Cp, double Cn, double B, double eps,
		   AdaptSolutionInfo* si, int shrinking);
	
protected:
	int active_size;
	bool is_auto_weight; 
	schar *y;
	double *G;		// gradient of objective function
	enum { LOWER_BOUND, UPPER_BOUND, FREE };
	char *alpha_status;	// LOWER_BOUND, UPPER_BOUND, FREE
	double *alpha;		
	const QMatrix *Q;
	const QMatrix *F;
	const Qfloat *QD;
	const Qfloat *FD;
	double eps;
	double Cp,Cn;	
	double B;
	int l;		//the number of data
	int	na;		//the number of aux model
	double *aux_fx;	
	
	int *active_set;
	double *G_bar;		// gradient, if we treat free variables as 0
	bool unshrinked;	// XXX

	double get_C(int i)
	{
		return (y[i] > 0)? Cp : Cn;
	}
	void update_alpha_status(int i)
	{
		if(alpha[i] >= get_C(i))
			alpha_status[i] = UPPER_BOUND;
		else if(alpha[i] <= 0)
			alpha_status[i] = LOWER_BOUND;
		else alpha_status[i] = FREE;
	}
	bool is_upper_bound(int i) { return alpha_status[i] == UPPER_BOUND; }
	bool is_lower_bound(int i) { return alpha_status[i] == LOWER_BOUND; }
	bool is_free(int i) { return alpha_status[i] == FREE; }
	void swap_index(int i, int j);
	void reconstruct_gradient_w();
	void reconstruct_gradient();
	virtual int select_working_set(int &out);
	virtual int max_violating_pair(int &i, int &j);
	virtual void do_shrinking();
};

void AdaptSolver::swap_index(int i, int j)
{
	Q->swap_index(i,j);
	if(is_auto_weight)
		F->swap_index(i,j);
	else
		swap(aux_fx[i],aux_fx[j]);
	swap(y[i],y[j]);
	swap(G[i],G[j]);
	swap(alpha_status[i],alpha_status[j]);
	swap(alpha[i],alpha[j]);
	swap(active_set[i],active_set[j]);
	swap(G_bar[i],G_bar[j]);	
}

// reconstruct inactive elements of G from G_bar and free variables
void AdaptSolver::reconstruct_gradient()
{	
	//quit if there is no inactive elements
	if(active_size == l) return;

	//only the Gi for inactive elements i has been modified,
	//the Gi for active elements remain the same
	int i;
	for(i=active_size;i<l;i++)	//traverse inactive elements
		G[i] = G_bar[i] - aux_fx[i];
	
	for(i=0;i<active_size;i++)	
		if(is_free(i))
		{
			const Qfloat *Q_i = Q->get_Q(i,l);
			double alpha_i = alpha[i];
			for(int j=active_size;j<l;j++) //traverse inactive elements
				G[j] += (alpha_i * Q_i[j] + alpha_i * y[i] * y[j]) ;
		}
}

// reconstruct inactive elements of G from G_bar and free variables
void AdaptSolver::reconstruct_gradient_w()
{	
	//quit if there is no inactive elements
	if(active_size == l) return;

	//only the Gi for inactive elements i has been modified,
	//the Gi for active elements remain the same
	int i;
	for(i=active_size;i<l;i++)	//traverse inactive elements
		G[i] = G_bar[i] - 1;
	
	for(i=0;i<active_size;i++)
		if(is_free(i))
		{
			const Qfloat *Q_i = Q->get_Q(i,l);
			const Qfloat *F_i = F->get_Q(i,l);
			double alpha_i = alpha[i];
			for(int j=active_size;j<l;j++) //traverse inactive elements
				G[j] += (alpha_i * Q_i[j] + (alpha_i * F_i[j] / B) + alpha_i * y[i] * y[j]);
		}
}

void AdaptSolver::Solve_w(const QMatrix& Q, const QMatrix& F, const schar *y_,
		   double *alpha_, double Cp, double Cn, double B, double eps, AdaptSolutionInfo* si, int shrinking)
{
	this->Q = &Q;
	this->F = &F;
	QD = Q.get_QD();
	FD = F.get_QD();
	
	clone(y, y_,l);	
	clone(alpha,alpha_,l);
	this->Cp = Cp;
	this->Cn = Cn;
	this->B = B;
	this->eps = eps;
	unshrinked = false;

	// initialize alpha_status
	{
		alpha_status = new char[l];
		for(int i=0;i<l;i++)
			update_alpha_status(i);	//whether each alpha is upperbound or lowerbound
	}

	// initialize active set (for shrinking)
	{
		active_set = new int[l];
		for(int i=0;i<l;i++)
			active_set[i] = i;	//at beginning, every instance is in the active set
		active_size = l;
	}

	// initialize gradient
	{
		//G_i = L_D '(alpha_i) = yi*f(xi) - 1 
		//= (yi/B) * \sum_j(alpha_j * yj * \sum_k (fk_j * fk_i))  + yi \sum_j (alpha_j * yj * K_ij) + yi * \sum_j(alpha_j * yj) - 1 
		//note this L_D is the negative of the L_D appearing in typical SVM derivation, and this L_D is to be minimized
		G = new double[l];
		G_bar = new double[l];
		int i;
		for(i=0;i<l;i++)
		{
			G[i] =  -1;
			G_bar[i] = 0;
		}
		for(i=0;i<l;i++)
			if(!is_lower_bound(i))	//if alpha_i = 0, i has no contribution to G
			{
				const Qfloat *Q_i = Q.get_Q(i,l);	//a row of the Q matrix (Q_ij = y_i * y_j * K_ij)
				const Qfloat *F_i = F.get_Q(i,l);	//a row of the F matrix (F_ij = y_i * y_j * \sum_k (fk_i * fk_j))
				double alpha_i = alpha[i];
				int j;
				for(j=0;j<l;j++)
					G[j] += alpha_i * Q_i[j] + (alpha_i * F_i[j] / B) + y[j] * alpha_i * y[i];
				
				if(is_upper_bound(i))	//alpha_i = C
					for(j=0;j<l;j++)
						G_bar[j] += get_C(i) * Q_i[j] + (get_C(i) * F_i[j] / B ) + y[j] * get_C(i) * y[i];  //get_C(i) * Q_i[j];
			}
	}

	// optimization step
	int iter = 0;
	int counter = min(5 * l,1000)+1;

	while(1)
	{
		// show progress and do shrinking when the current counter expires
		// it doesn't stop the iteration since a new counter is set
		if(--counter == 0)
		{
			counter = min(5 * l,1000);
			if(shrinking) do_shrinking(); 
			info("."); info_flush();
		}

		int i /*,j */;
		
		//if cannot find a working variable to improve the obj function (when return  = 1), 
		//we reconstruct the gradient and try again; if fails agin, quick, otherwise,
		//continue and do shrinkage in next iteration
		if(select_working_set(i)!=0)		
		{
			// reconstruct the whole gradient
			reconstruct_gradient_w();
			// reset active set size and check
			active_size = l;
			info("*"); info_flush();
			if(select_working_set(i)!=0)
				break;
			else
				counter = 1;	// do shrinking next iteration
		}
		
		++iter;

		// update alpha[i] and handle bounds carefully			
		const Qfloat *Q_i = Q.get_Q(i,active_size);
		const Qfloat *F_i = F.get_Q(i,active_size);
		double C_i = get_C(i);
		double old_alpha_i = alpha[i];
		alpha[i] = old_alpha_i - (G[i] / (Q_i[i] + (F_i[i] / B) + 1));

		if(alpha[i] < 0) alpha[i] = 0;
		if(alpha[i] > C_i) alpha[i] = C_i;

		double delta_alpha_i = alpha[i] - old_alpha_i;

		// update G
		for(int k=0;k<active_size;k++)
		{
			G[k] += (Q_i[k] * delta_alpha_i + (F_i[k] * delta_alpha_i / B) + y[k] * delta_alpha_i * y[i]);
		}

		// update alpha_status and G_bar
		{
			//read the old status
			bool ui = is_upper_bound(i);
			update_alpha_status(i);
			
			int k;
			if(ui != is_upper_bound(i))	//upper bound status changes for i
			{
				Q_i = Q.get_Q(i,l);
				F_i = F.get_Q(i,l);
				
				if(ui)
				{
					//upper bound -> no upper bound
					for(k=0;k<l;k++)
						G_bar[k] -= (C_i * Q_i[k] +(C_i * F_i[k] / B) + y[k] * C_i * y[i]);
				}
				else
				{
					//no upper bound -> upper bound
					for(k=0;k<l;k++)
						G_bar[k] += (C_i * Q_i[k] + (C_i * F_i[k] / B) + y[k] * C_i * y[i]); 
				}
			}
		}
	}

	// calculate threshold (only happens after all SMO iterations)
	//si->rho = calculate_rho();
	si->rho = 0;
	for(int k=0; k<l; k++)
		si->rho -= alpha[k] * y[k];
	
	// calculate objective value
	{
		double v = 0;
		int i;
		for(i=0;i<l;i++)
			v += alpha[i] * (G[i] - 1);

		si->obj = v/2;
	}

	// put back the solution
	{
		for(int i=0;i<l;i++)
			alpha_[active_set[i]] = alpha[i];
	}

	si->upper_bound_p = Cp;
	si->upper_bound_n = Cn;

	info("\noptimization finished, #iter = %d\n",iter);

	delete[] y;
	delete[] alpha;
	delete[] alpha_status;
	delete[] active_set;
	delete[] G;
	delete[] G_bar;
}

//Jun Yang: we change the original arugment "b" to "lambda" for adaptation
void AdaptSolver::Solve(const QMatrix& Q, const double *aux_fx_, const schar *y_,
		   double *alpha_, double Cp, double Cn, double eps,
		   AdaptSolutionInfo* si, int shrinking)
{
	this->Q = &Q;
	QD=Q.get_QD();
	clone(y, y_,l);
	clone(aux_fx, aux_fx_,l);
	clone(alpha,alpha_,l);
	this->Cp = Cp;
	this->Cn = Cn;
	this->eps = eps;
	unshrinked = false;

	// initialize alpha_status
	{
		alpha_status = new char[l];
		for(int i=0;i<l;i++)
			update_alpha_status(i);	//whether each alpha is upperbound or lowerbound
	}

	// initialize active set (for shrinking)
	{
		active_set = new int[l];
		for(int i=0;i<l;i++)
			active_set[i] = i;	//at beginning, every instance is in the active set
		active_size = l;
	}

	// initialize gradient
	{
		//G_i = L_D '(alpha_i) = yi*f(xi) - 1 = - aux_fx_i + y_i * \sum_j(alpha_j*y_j) + y_i \sum_j (alpha_j * y_j * K_ij) 
		//aux_fx = 1 - y_i * f^a(x_i)		
		//note this L_D is the negative of the L_D appearing in typical SVM derivation, and this L_D is to be minimized
		G = new double[l];
		G_bar = new double[l];
		int i;
		for(i=0;i<l;i++)
		{
			G[i] =  - aux_fx[i];
			G_bar[i] = 0;
		}
		for(i=0;i<l;i++)
			if(!is_lower_bound(i))	//if alpha_i = 0, i has no contribution to G
			{
				const Qfloat *Q_i = Q.get_Q(i,l);	//a row of the Q matrix (Q_ij = y_i * y_j * K_ij)
				double alpha_i = alpha[i];
				int j;
				for(j=0;j<l;j++)
					G[j] += alpha_i * Q_i[j] + y[j] * alpha_i * y[i];   //alpha_i*Q_i[j];
				
				if(is_upper_bound(i))
					for(j=0;j<l;j++)
						G_bar[j] += get_C(i) * Q_i[j] + y[j] * get_C(i) * y[i];  //get_C(i) * Q_i[j];
			}
	}

	// optimization step
	int iter = 0;
	int counter = min(5 * l,1000)+1;

	while(1)
	{
		// show progress and do shrinking when the current counter expires
		// it doesn't stop the iteration since a new counter is set
		if(--counter == 0)
		{
			counter = min(5 * l,1000);
			if(shrinking) do_shrinking(); 
			info("."); info_flush();
		}

		int i /*,j */;
		
		//if cannot find a working variable to improve the obj function (when return  = 1), 
		//we reconstruct the gradient and try again; if fails agin, quick, otherwise,
		//continue and do shrinkage in next iteration
		if(select_working_set(i)!=0)		
		{
			// reconstruct the whole gradient
			reconstruct_gradient();
			// reset active set size and check
			active_size = l;
			info("*"); info_flush();
			if(select_working_set(i)!=0)
				break;
			else
				counter = 1;	// do shrinking next iteration
		}
		
		++iter;

		// update alpha[i] and handle bounds carefully			
		const Qfloat *Q_i = Q.get_Q(i,active_size);
		double C_i = get_C(i);
		double old_alpha_i = alpha[i];
		alpha[i] = old_alpha_i - (G[i] / (Q_i[i] + 1));

		if(alpha[i] < 0) alpha[i] = 0;
		if(alpha[i] > C_i) alpha[i] = C_i;

		double delta_alpha_i = alpha[i] - old_alpha_i;

		// update G
		for(int k=0;k<active_size;k++)
		{
			G[k] += (Q_i[k] * delta_alpha_i + y[k] * delta_alpha_i * y[i]);
		}

		// update alpha_status and G_bar
		{
			//read the old status
			bool ui = is_upper_bound(i);
			update_alpha_status(i);
			
			int k;
			if(ui != is_upper_bound(i))	//upper bound status changes for i
			{
				Q_i = Q.get_Q(i,l);
				
				if(ui)
				{
					//upper bound -> no upper bound
					for(k=0;k<l;k++)
						G_bar[k] -= (C_i * Q_i[k] + y[k] * C_i * y[i]);
				}
				else
				{
					//no upper bound -> upper bound
					for(k=0;k<l;k++)
						G_bar[k] += (C_i * Q_i[k] + y[k] * C_i * y[i]); 
				}
			}
		}
	}

	// calculate threshold (only happens after all SMO iterations)
	//  = calculate_rho();
	si->rho = 0;
	for(int k=0; k<l; k++)
		si->rho -= alpha[k] * y[k];

	// calculate objective value
	{
		double v = 0;
		int i;
		for(i=0;i<l;i++)
			v += alpha[i] * (G[i] - aux_fx[i]);

		si->obj = v/2;
	}

	// put back the solution
	{
		for(int i=0;i<l;i++)
			alpha_[active_set[i]] = alpha[i];
	}

	si->upper_bound_p = Cp;
	si->upper_bound_n = Cn;

	info("\noptimization finished, #iter = %d\n",iter);

	delete[] aux_fx;
	delete[] y;
	delete[] alpha;
	delete[] alpha_status;
	delete[] active_set;
	delete[] G;
	delete[] G_bar;
}


// return 1 if already optimal, return 0 otherwise
int AdaptSolver::select_working_set(int &out)
{
	double Gmin = 0;
	double Gmax = 0;
	int Gmin_idx = -1;
	int Gmax_idx = -1;

	for(int t=0;t<active_size;t++)
	{
		if(!is_upper_bound(t))
		{
			if(G[t] <= Gmin)
			{
				Gmin = G[t];
				Gmin_idx = t;
			}
		}

		if(!is_lower_bound(t))
		{
			if(G[t] >= Gmax)
			{
				Gmax = G[t];
				Gmax_idx = t;
			}
		}
	}	
	
	if(Gmax - Gmin < eps)
		return 1;

	if(abs(Gmax) >= abs(Gmin))
		out = Gmax_idx;
	else 
		out = Gmin_idx;
	return 0;
}

// return 1 if already optimal, return 0 otherwise
int AdaptSolver::max_violating_pair(int &out_i, int &out_j)
{
	
	double Gmin = 0;
	double Gmax = 0;
	out_i = -1;
	out_j = -1;
	
	for(int t=0;t<active_size;t++)
	{
		if(!is_upper_bound(t))
		{
			if(G[t] <= Gmin)
			{
				Gmin = G[t];
				out_i = t;
			}
		}

		if(!is_lower_bound(t))
		{
			if(G[t] >= Gmax)
			{
				Gmax = G[t];
				out_j = t;
			}
		}
	}
		
	if(Gmax - Gmin < eps)
		return 1;

	return 0;

}

void AdaptSolver::do_shrinking()
{
	int i,j,k;	
	if(max_violating_pair(i,j)!=0) return;

	double Gmin = 0;
	if(i != -1) Gmin = G[i];
	double Gmax = 0;
	if(j != -1) Gmax = G[j];
    
	for(k=0;k<active_size;k++)
	{
		if(is_lower_bound(k))
		{
			//if(G[k] <= Gmin)
			if(G[k] <= Gmax)
				continue;		//continue if active
		}
		else if(is_upper_bound(k))
		{
			//if(G[k] >= Gmax)
			if(G[k] >= Gmin)
				continue;			
		}
		else 
			continue;

		//if runs to this point, k must be inactive, so we swap it outside active set
		--active_size;
		swap_index(k,active_size);
		--k;	// look at the newcomer
	}
	
	//if -(Gm1 + Gm2)  = m - M >= 10*eps ==> m <= M + 10*eps 
	// ==> reconstrunct the gradient to increase accuracy
	if(unshrinked || Gmax - Gmin > eps*10) return;
	
	unshrinked = true;
	if(is_auto_weight)
		reconstruct_gradient_w();
	else
		reconstruct_gradient();

	for(k=l-1;k>=active_size;k--)
	{
		if(is_lower_bound(k))
		{
			//if(-G[k] > Gmin) continue;	//continue if inactive
			if(G[k] > Gmax) continue;	//continue if inactive
		}
		else if(is_upper_bound(k))
		{
			//if(G[k] < Gmax)	continue;	//continue if inactive
			if(G[k] < Gmin)	continue;	//continue if inactive				
		}
		else continue;

		swap_index(k,active_size);
		active_size++;
		++k;	// look at the newcomer
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//other functions


static void adapt_solve_c_svc(const svm_problem *prob, const svm_parameter* param, 
							  struct svm_model** aux_models, double *alpha, AdaptSolver::AdaptSolutionInfo* si, double Cp, double Cn)
{
	int l = prob->l;
	schar *y = new schar[l];

	int i;
	//initialize the model parameters
	for(i=0;i<l;i++)
	{
		alpha[i] = 0;
		if(prob->y[i] > 0) y[i] = +1; else y[i]=-1;
	}

	double *aux_fx;
	if(param->is_auto_weight)
	{
		//automatically weight multiple auxiliary models
		aux_fx = Malloc(double,prob->l * param->num_aux);
		aux_model_predictions(prob, param, aux_models, aux_fx);	

		AdaptSolver s(true, l, param->num_aux);
		s.Solve_w(SVC_Q(*prob,*param,y), SVC_Q(*prob,*param,y,aux_fx), y, alpha, Cp, Cn, param->B, param->eps, si, param->shrinking);
	}
	else
	{
		//use average or manually assigned weights for auxiliary models
		aux_fx = Malloc(double,prob->l);
		aux_model_predictions_ensemble(prob, param, aux_models, aux_fx);	

		AdaptSolver s(false, l, param->num_aux);
		s.Solve(SVC_Q(*prob,*param,y), aux_fx, y, alpha, Cp, Cn, param->eps, si, param->shrinking);
	}

	double sum_alpha=0;
	for(i=0;i<l;i++)
		sum_alpha += alpha[i];

	if (Cp==Cn)
		info("nu = %f\n", sum_alpha/(Cp*prob->l));

	//after the model is trained, we let alpha to absorb y, 
	//so that we don't need to save both alpha and y in the model file		
	for(i=0;i<l;i++)
		alpha[i] *= y[i];
	
	if(param->is_auto_weight)
	{
		for(int k=0; k<param->num_aux; k++)
			param->aux_weights[k] = 0;
	
		for(i=0;i<l;i++)
		{
			for(int k=0; k<param->num_aux; k++)
				param->aux_weights[k] += alpha[i] * aux_fx[k*l + i] / param->B;
		}
	}
	
	delete[] y;
}

// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
void svm_group_classes(const svm_problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
	int l = prob->l;
	int max_nr_class = 16;
	int nr_class = 0;
	int *label = Malloc(int,max_nr_class);
	int *count = Malloc(int,max_nr_class);
	int *data_label = Malloc(int,l);	
	int i;

	for(i=0;i<l;i++)
	{
		int this_label = (int)prob->y[i];
		int j;
		for(j=0;j<nr_class;j++)
		{
			if(this_label == label[j])
			{
				++count[j];
				break;
			}
		}
		data_label[i] = j;
		if(j == nr_class)
		{
			if(nr_class == max_nr_class)
			{
				max_nr_class *= 2;
				label = (int *)realloc(label,max_nr_class*sizeof(int));
				count = (int *)realloc(count,max_nr_class*sizeof(int));
			}
			label[nr_class] = this_label;
			count[nr_class] = 1;
			++nr_class;
		}
	}

	int *start = Malloc(int,nr_class);
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[data_label[i]]] = i;
		++start[data_label[i]];
	}
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];

	*nr_class_ret = nr_class;
	*label_ret = label;
	*start_ret = start;
	*count_ret = count;
	free(data_label);
}

decision_function adapt_svm_train_one(const svm_problem *prob, const svm_parameter *param, 
									  struct svm_model** aux_models,double Cp, double Cn)
{
	double *alpha = Malloc(double,prob->l);	

	AdaptSolver::AdaptSolutionInfo si;
	adapt_solve_c_svc(prob,param, aux_models, alpha,&si,Cp,Cn);
	
	info("obj = %f, rho = %f\n",si.obj,si.rho);

	// output SVs
	int nSV = 0;
	int nBSV = 0;
	for(int i=0;i<prob->l;i++)
	{
		if(fabs(alpha[i]) > 0)
		{
			++nSV;
			if(prob->y[i] > 0)
			{
				if(fabs(alpha[i]) >= si.upper_bound_p)
					++nBSV;
			}
			else
			{
				if(fabs(alpha[i]) >= si.upper_bound_n)
					++nBSV;
			}
		}
	}

	info("nSV = %d, nBSV = %d\n",nSV,nBSV);

	decision_function f;
	f.alpha = alpha;
	f.rho = si.rho;
	return f;
}

svm_model *adapt_svm_train(const svm_problem *prob, const svm_parameter *param, struct svm_model** aux_models)
{
	svm_model *model = Malloc(svm_model,1);
	model->param = *param;
	model->free_sv = 0;	

	//adaptive SVM can only handle classification task
	if(param->svm_type != C_SVC)
	{
		printf("Adpative SVM can only handle classification task! Quit...\n");
		exit(1);
	}		

	// classification
	int l = prob->l;
	int nr_class;
	int *label = NULL;
	int *start = NULL;
	int *count = NULL;
	int *perm = Malloc(int,l);

	// group training data of the same class
	svm_group_classes(prob,&nr_class,&label,&start,&count,perm);
	if(nr_class != 2)
	{
		printf("Adpative SVM can only handle binary classification! Quit...\n");
		exit(1);
	}

	//if the order of labels are different, change the order in auxiliary model s.t. it is
	//consistent with the order in the current problem
	for(int k=0; k<param->num_aux; k++)
	{
		if(label[0] != aux_models[k]->label[0])
		{
			swap(aux_models[k]->label[0], aux_models[k]->label[1]);
			swap(aux_models[k]->nSV[0], aux_models[k]->nSV[1]);
			
			for(int i=0; i<aux_models[k]->l; i++)
				aux_models[k]->sv_coef[0][i] = -aux_models[k]->sv_coef[0][i];
			
			aux_models[k]->rho[0] = - aux_models[k]->rho[0];	//assuming 2-class case
		}
	}
	
	svm_node **x = Malloc(svm_node *,l);
	int i;
	for(i=0;i<l;i++)
		x[i] = prob->x[perm[i]];

	// calculate weighted C
	double *weighted_C = Malloc(double, nr_class);
	for(i=0;i<nr_class;i++)
		weighted_C[i] = param->C;
	for(i=0;i<param->nr_weight;i++)
	{	
		int j;
		for(j=0;j<nr_class;j++)
			if(param->weight_label[i] == label[j])
				break;
		if(j == nr_class)
			fprintf(stderr,"warning: class label %d specified in weight is not found\n", param->weight_label[i]);
		else
			weighted_C[j] *= param->weight[i];
	}

	// train k*(k-1)/2 models		
	bool *nonzero = Malloc(bool,l);
	for(i=0;i<l;i++)
		nonzero[i] = false;
	decision_function *f = Malloc(decision_function,nr_class*(nr_class-1)/2);

	double *probA=NULL,*probB=NULL;

	int p = 0;
	for(i=0;i<nr_class;i++)
		for(int j=i+1;j<nr_class;j++)
		{
			svm_problem sub_prob;
			int si = start[i], sj = start[j];
			int ci = count[i], cj = count[j];
			sub_prob.l = ci+cj;
			sub_prob.x = Malloc(svm_node *,sub_prob.l);
			sub_prob.y = Malloc(double,sub_prob.l);
			int k;
			for(k=0;k<ci;k++)
			{
				sub_prob.x[k] = x[si+k];
				sub_prob.y[k] = +1;
			}
			for(k=0;k<cj;k++)
			{
				sub_prob.x[ci+k] = x[sj+k];
				sub_prob.y[ci+k] = -1;
			}

			f[p] = adapt_svm_train_one(&sub_prob, param, aux_models, weighted_C[i],weighted_C[j]);
			for(k=0;k<ci;k++)
				if(!nonzero[si+k] && fabs(f[p].alpha[k]) > 0)
					nonzero[si+k] = true;
			for(k=0;k<cj;k++)
				if(!nonzero[sj+k] && fabs(f[p].alpha[ci+k]) > 0)
					nonzero[sj+k] = true;
			free(sub_prob.x);
			free(sub_prob.y);
			++p;
		}

	// build output
	model->nr_class = nr_class;
	
	model->label = Malloc(int,nr_class);
	for(i=0;i<nr_class;i++)
		model->label[i] = label[i];
	
	model->rho = Malloc(double,nr_class*(nr_class-1)/2);
	for(i=0;i<nr_class*(nr_class-1)/2;i++)
	{
		model->rho[i] = f[i].rho;

		for(int k = 0; k<param->num_aux; k++)
			model->rho[i] += aux_models[k]->rho[i];
	}

	//probabilistic output not supported yet
	model->probA=NULL;
	model->probB=NULL;

	int total_sv = 0;
	int *nz_count = Malloc(int,nr_class);
	model->nSV = Malloc(int,nr_class);
	for(i=0;i<nr_class;i++)
	{
		int nSV = 0;
		for(int j=0;j<count[i];j++)
			if(nonzero[start[i]+j])
			{	
				++nSV;
				++total_sv;
			}
		model->nSV[i] = nSV;
		nz_count[i] = nSV;
	}
	
	info("Total nSV = %d\n",total_sv);

	model->l = total_sv;
	model->SV = Malloc(svm_node *,total_sv);
	p = 0;
	for(i=0;i<l;i++)
		if(nonzero[i]) model->SV[p++] = x[i];

	int *nz_start = Malloc(int,nr_class);
	nz_start[0] = 0;
	for(i=1;i<nr_class;i++)
		nz_start[i] = nz_start[i-1]+nz_count[i-1];

	model->sv_coef = Malloc(double *,nr_class-1);
	for(i=0;i<nr_class-1;i++)
		model->sv_coef[i] = Malloc(double,total_sv);

	p = 0;
	for(i=0;i<nr_class;i++)
		for(int j=i+1;j<nr_class;j++)
		{
			// classifier (i,j): coefficients with
			// i are in sv_coef[j-1][nz_start[i]...],
			// j are in sv_coef[i][nz_start[j]...]

			int si = start[i];
			int sj = start[j];
			int ci = count[i];
			int cj = count[j];
			
			int q = nz_start[i];
			int k;
			for(k=0;k<ci;k++)
				if(nonzero[si+k])
					model->sv_coef[j-1][q++] = f[p].alpha[k];
			q = nz_start[j];
			for(k=0;k<cj;k++)
				if(nonzero[sj+k])
					model->sv_coef[i][q++] = f[p].alpha[ci+k];
			++p;
		}
	
	free(label);
	free(probA);
	free(probB);
	free(count);
	free(perm);
	free(start);
	free(x);
	free(weighted_C);
	free(nonzero);
	for(i=0;i<nr_class*(nr_class-1)/2;i++)
		free(f[i].alpha);
	free(f);
	free(nz_count);
	free(nz_start);
	
	return model;
}
