#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "svm.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

void exit_with_help()
{
	printf(
		"Usage: adapt-svm-train [options] [+ source_model_file[,weight]] train_data_file [target_model_file]\n"
	"options:\n"	
	"-t kernel_type : set type of kernel function (default 2)\n"
	"	0 -- linear: u'*v\n"
	"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
	"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
	"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
	"	4 -- precomputed kernel (kernel values in training_set_file)\n"
	"-d degree : set degree in kernel function (default 3)\n"
	"-g gamma : set gamma in kernel function (default 1/k)\n"
	"-c cost : set the parameter C (default 1)\n"
	"-m cachesize : set cache memory size in MB (default 100)\n"
	"-e epsilon : set tolerance of termination criterion (default 0.001)\n"
	"-h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)\n"	
	"-wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)\n"	
	"-l learn model weights: whether to learn weights of source models (default 0)\n"	
	"-s contribution of source model : effective only when -l 1 (default 1)\n"	
	"\nexample:\n adapt-svm-train -t 2 -k 1 -b 10 + model1.txt,0.8 + model2.txt,0.2 train.txt\n"
	);
	exit(1);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name, char **aux_model_file_names);
void read_problem(const char *filename);
struct svm_parameter param;		// set by parse_command_line
struct svm_problem prob;		// set by read_problem
struct svm_model *model;
struct svm_model** aux_models;	//Jun Yang: added to represent the auxiliary model
struct svm_node *x_space;
typedef void* pt_svm_model;

const int MAX_AUX = 32;

int main(int argc, char **argv)
{
	char input_file_name[1024];
	char* aux_model_file_names[MAX_AUX];
	char model_file_name[1024];
	const char *error_msg;

	parse_command_line(argc, argv, input_file_name, model_file_name, aux_model_file_names);
	read_problem(input_file_name);

	//read in the auxiliary models
	aux_models = Malloc(svm_model*, param.num_aux);
	
	for(int i=0; i< param.num_aux; i++)
		if((aux_models[i] = svm_load_model(aux_model_file_names[i])) == 0){
			fprintf(stderr,"can't open auxiliary model file %s\n",aux_model_file_names[i]);
			exit(1);
		}		

	error_msg = svm_check_parameter(&prob,&param);
	if(error_msg)
	{
		fprintf(stderr,"Error: %s\n",error_msg);
		exit(1);
	}

	//We check whether the specified parameters are consistent with the parameters of each auxiliary model.	
	for(int i=0; i<param.num_aux; i++)
	{		
		if(param.svm_type != aux_models[i]->param.svm_type
			|| param.kernel_type != aux_models[i]->param.kernel_type 		
			|| aux_models[i]->nr_class != 2) //currently we only handle 2-class problem
		{
			fprintf(stderr, "Error: The specified parameters are not consistent with those in the auxiliary model!\n");
			exit(1);
		}
	}

	//train the model
	model = adapt_svm_train(&prob, &param, aux_models);	
	adapt_svm_save_model(model_file_name, model, aux_models);
	
	//free the space
	svm_destroy_model(model);	
	svm_destroy_param(&param);
	free(prob.y);
	free(prob.x);
	free(x_space);
	for(int i = 0; i< param.num_aux; i++)
		free(aux_model_file_names[i]);

	return 0;
}

//Jun Yang: I modified this function from Libsvm, so that it also 
//reads the filename of an auxiliary model
void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name, char **aux_model_file_names)
{
	int i;

	// default values	
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 0;	// 1/k
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 1;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
    
	//initialization for parameters related to adaptaiton
	param.num_aux = 0;
	param.is_auto_weight = false;
	param.B = 1;
	param.aux_weights = Malloc(double, MAX_AUX);
	//cross_validation = 0;

	// parse options	
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-' && argv[i][0] != '+') break;
		if(++i>=argc)
			exit_with_help();

		if(argv[i-1][0] == '-')
		{
			switch(argv[i-1][1])
			{
			case 'l':
				param.is_auto_weight = (atoi(argv[i]) != 0);
				break;
			case 't':
				param.kernel_type = atoi(argv[i]);
				break;
			case 'd':
				param.degree = atoi(argv[i]);
				break;
			case 'g':
				param.gamma = atof(argv[i]);
				break;
			case 'r':
				param.coef0 = atof(argv[i]);
				break;
			case 'm':
				param.cache_size = atof(argv[i]);
				break;
			case 'c':
				param.C = atof(argv[i]);
				break;
			case 's':
				param.B = 1 / atof(argv[i]);
				break;
			case 'e':
				param.eps = atof(argv[i]);
				break;
			case 'h':
				param.shrinking = atoi(argv[i]);
				break;
			case 'w':
				++param.nr_weight;
				param.weight_label = (int *)realloc(param.weight_label,sizeof(int)*param.nr_weight);
				param.weight = (double *)realloc(param.weight,sizeof(double)*param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
				param.weight[param.nr_weight-1] = atof(argv[i]);
				break;
			default:
				fprintf(stderr,"unknown option\n");
				exit_with_help();
			}
		}
		else
		{
			//'+' for auxiliary model
			char* pos =  strrchr(argv[i], ',');
			if(pos){
				aux_model_file_names[param.num_aux] = Malloc(char, (pos - argv[i] + 1));
				strncpy(aux_model_file_names[param.num_aux], argv[i], (pos - argv[i]));
				aux_model_file_names[param.num_aux][pos-argv[i]] = '\0';
				param.aux_weights[param.num_aux++] = atof(pos+1);				
			}else{
				aux_model_file_names[param.num_aux] = Malloc(char, strlen(argv[i])+1);
				strcpy(aux_model_file_names[param.num_aux], argv[i]);
				param.aux_weights[param.num_aux++] = 1;
			}
		}
	}

	// determine filenames
	if(i>=argc)
		exit_with_help();
	
	strcpy(input_file_name, argv[i]);
	if(i<argc-1)
		strcpy(model_file_name,argv[i+1]);
	else
	{
		char *p = strrchr(input_file_name,'/');
		if(p==NULL)
			p = input_file_name;
		else
			++p;
		sprintf(model_file_name,"%s.model",p);
	}	
}

// read in a problem (in svmlight format)
void read_problem(const char *filename)
{
	int elements, max_index, i, j;
	FILE *fp = fopen(filename,"r");
	
	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	prob.l = 0;
	elements = 0;
	while(1)
	{
		int c = fgetc(fp);
		switch(c)
		{
			case '\n':
				++prob.l;
				// fall through,
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
	rewind(fp);

	//prob.l is the total number of data points,
	//while elements is the total number of feature components 
	//in the whole dataset PLUS the total number of data points
	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct svm_node *,prob.l);
	x_space = Malloc(struct svm_node,elements);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		double label;
		prob.x[i] = &x_space[j];
		fscanf(fp,"%lf",&label);
		prob.y[i] = label;

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
		if(j>=1 && x_space[j-1].index > max_index)
			max_index = x_space[j-1].index;
		x_space[j++].index = -1;
	}

	if(param.gamma == 0)
		param.gamma = 1.0/max_index;

	if(param.kernel_type == PRECOMPUTED)
		for(i=0;i<prob.l;i++)
		{
			if (prob.x[i][0].index != 0)
			{
				fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
				exit(1);
			}
			if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			{
				fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
				exit(1);
			}
		}

	fclose(fp);
}
