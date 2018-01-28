/*************************************************************************/
/* This program has two usages:
*
* Training: <program> -t data_filename tree_filename
*         will grow a tree from the data and store it in a file
*
* Prediction: <program> -p data_filename tree_filename
*         will read a tree from a file and predict the value of Y for the data
*
* Bernard Merialdo, Benoit Huet, Emilie Dumont 1999-2016
*/
/*************************************************************************/
#define _CRT_SECURE_NO_DEPRECATE 

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

// STL library includes
#include <fstream>
#include <iostream>
#include <vector>
#include <set>
using namespace std;
/*************************************************************************/
/* 
*   These global variables hold the training data.
*
*   We try to predict the value of Y given the values of X[0], X[1]... X[nX-1]
*   Y[i] = 0 if the category of input data sample i is equal to category 
*   Y[i] = 1 if the category of input data sample i is NOT equal to 
*   category beeing predicted
*   X[j][i] is the value of attribute j of data sample i
*
*   The training data is a set of nT samples:
*   Y[i] X[0][i] X[1][i] ... X[nX-1][i], i=0, 1, ... nT-1
*
*   which compose a rectangular matrix:
*   Y[0] X[0][0] X[1][0] ... X[nX-1][0]
*   Y[1] X[0][1] X[1][1] ... X[nX-1][1]
*   ...
*   Y[nT-1] X[0][nT-1] X[1][nT-1] ... X[nX-1][nT-1]
*/
/*************************************************************************/
int nX;
int nT;
vector<int> Y;
vector<vector<float> > X;
vector<vector<float> > values;

struct Node {
	int number;			// the node number
	int depth;			// the depth of this node within the tree (0 is the root node)
	int category;		// the category Y[]==0 or Y[]==1 at the node
	int attribute;		// the attribute X[attribute][] on which the decision at this node is made
	float threshold;		// the threshold used for this decision
	Node *son1;			// pointer to the tree node for which X[][] <= threshold
	Node *son2;			// pointer to the tree node for which X[][] > threshold
	float falseNb;		// the number of wrong number of evaluations
	float totalNb;		// the total number of evaluations
	float entropy;		// the entropy at this node.
};

int max_depth; // the maximum depth allowed in the tree

Node *root_node;    // the root node of the tree (should also be equal tree[0])
int treeSize; // the number of nodes in the tree
float Log2 = (float)log(2.); //Log2 is just a constant used to compute Log(p) in base 2

/*************************************************************************/
// this procedure reads the file "filename", computes the number
// of attributes nX and the number of samples nT, allocates the
// arrays X and Y, and fills the values of these arrays from the file
void Read_data(char *filename, char *reference) {
	FILE *in; 
	int i;
	char c;
	char Yname[256];
	char Xname[256];
	int j=0;

	// open input file 
	if((in = fopen(filename,"r")) == NULL) {
		printf("*** Error opening file %s\n",filename);
		getchar();
		exit(1);
	}
	printf("Reading file %s\n",filename);

	// first read the file to compute ND and nX 
	nT = 0;
	nX = 0;
	while(fscanf(in,"%[^ \n\t]",Yname) > 0) {
		for(i=0; ; i++) {
			fscanf(in," %[^ \n\t]%c",Xname,&c);
			if(c == '\n') break;
		}
		if(nX == 0) nX = i+1;
		else if(nX != i+1) {
			printf("*** Error: wrong number of attributes in line %d: %d - %d\n",
				nT,nX,i+1);
			getchar();
			exit(1);
		}
		nT++;
	}

	// allocate arrays with proper dimensions 
	Y.resize(nT);
	X.resize(nX);
	for(i=0; i<nX; i++) X[i].resize(nT);

	// now read the file and fill in the arrays 
	rewind(in);
	nT = 0;
	while(fscanf(in,"%[^ \n\t]",Yname) > 0) {
		if(strcmp(Yname,reference) == 0) Y[nT] = 0;
		else Y[nT] = 1;

		for(i=0; i<nX; i++) {
			fscanf(in," %[^ \n\t]",Xname);
			X[i][nT] = (float)atof(Xname);
		}
		fscanf(in,"\n");
		nT++;
	}

	printf("   %d samples with %d attributes\n",nT,nX);
	fclose(in);

	// list the different values of X
	values.resize(nX);
	set<float> v;
	set<float>::iterator iv;
	for(i=0; i<nX; i++) {
		v.clear();
		for(j=0; j<nT; j++) {
			v.insert(X[i][j]);
		}
		values[i].resize(v.size());
		iv = v.begin();
		for(j=0; j<(int)v.size(); j++) {
			values[i][j] = *iv;
			iv++;
		}
	}
}

/*************************************************************************/
/*
* This procedure looks for the best split of the data for node "current_node"
* in two parts, based on the value of attributes.
* The data for current node is assumed to be in the arrays between
* indices first_sample and last_sample.
* The entropy of the current node is computed, then for every attribute
* and every possible threshold, the entropy of the split is computed.
* The best split is kept and applied if there is an entropy improvement.
* When a new split is introduced, the data is sorted so that the samples
* for a given node are contiguous, and the procedure is called recursively.
*/
void Split_node(Node * current_node, int depth, int first_sample, int last_sample) {
	int N, NY0, NY1;
	int N1, N1Y0, N1Y1;
	int N2, N2Y0, N2Y1;
	int i;
	int k;
	float p;
	float node_entropy;
	float average_entropy;
	float entropy1;
	float entropy2;
	float best_entropy;
	float threshold;
	float best_threshold;
	int iX;
	int best_iX;
	int y;
	float x;
	int ivalues; 
	clock_t last_clock = clock();

	// initialize
	current_node->depth = depth;
	current_node->attribute = -1;
	current_node->threshold = 0.;
	current_node->son1 = NULL;
	current_node->son2 = NULL;

	// compute the probability of the values of Y for this node 
	// p(Y = 0) = N(Y = 0) / N = NY0 / N
	// p(Y = 1) = N(Y = 1) / N = NY1 / N
	NY0 = 0;
	NY1 = 0;
	for(i = first_sample; i<= last_sample; i++) {
		if(Y[i] ==  0) NY0++;
		else NY1++;
	}
	N = NY0 + NY1;

	// compute node entropy 
	node_entropy = 0.;
	if(NY0 > 0) {
		p = (float)NY0 / (float) N;
		node_entropy -=  p * log(p) / Log2;
	}
	if(NY1 > 0) {
		p = (float)NY1 / (float) N;
		node_entropy -=  p * log(p) / Log2;
	}

	printf("Node %d depth=%d size=%d N(Y=0)=%d N(Y=1)=%d entropy=%g\n",current_node->number,depth,N,NY0,NY1,node_entropy);
	current_node->entropy = node_entropy;
	if(NY0<NY1) current_node->category = 1; // the question X[][]<= threshold gets more Y[]==1
	else current_node->category = 0; // the question X[][]<= threshold gets more Y[]==0

	if(depth < max_depth) {
		// find best attribute X and best threshold to split 
		best_entropy = FLT_MAX; // initial best entropy is very large 
		best_iX = -1;
		best_threshold = 0.;

		// try over all attributes 
		for(iX=0; iX<nX; iX++) {
			// try over all possible attributes thresholds 
			for(ivalues=0; ivalues<(int)values[iX].size(); ivalues++) {
				if(clock() > last_clock + 30*CLOCKS_PER_SEC) {
					printf("  ...trying attribute %d\n",iX); // display a message during long computations
					last_clock = clock();
				}
				/******************************************************************/
				/* BEGIN IMPLEMENTATION                                           */
				/******************************************************************/

				threshold = values[iX][ivalues];

				N1Y0 = 0; N1Y1 = 0; N2Y0 = 0; N2Y1 = 0;
				for (int i = first_sample; i <= last_sample; i++)
				{
					if (X[iX][i] <= threshold) {
						if (Y[i] == 0) {
							N1Y0 += 1;
						}
						else {
							N1Y1 += 1;
						}
						
					}
					else {
						if (Y[i] == 0) {
							N2Y0 += 1;
						}
						else {
							N2Y1 += 1;
						}
					}
				}

				N1 = N1Y0 + N1Y1;
				N2 = N2Y0 + N2Y1;

				entropy1 = 0.;
				if (N1Y0 > 0) {
					p = (float)N1Y0 / (float)N1;
					entropy1 -= p * log(p) / Log2;
				}
				if (N1Y1 > 0) {
					p = (float)N1Y1 / (float)N1;
					entropy1 -= p * log(p) / Log2;
				}
				// compute entropy2 = H(Y / X[iX] > threshold)
				entropy2 = 0;
				if (N2Y0 > 0) {
					p = (float)N2Y0 / (float)N2;
					entropy2 -= p * log(p) / Log2;
				}
				if (N2Y1 > 0) {
					p = (float)N2Y1 / (float)N2;
					entropy2 -= p * log(p) / Log2;
				}
				N = N1 + N2;
				average_entropy = (float)(N1*entropy1 + N2 *entropy2) / N;



				/******************************************************************/
				/* END IMPLEMENTATION                                             */
				/******************************************************************/

				// compare entropy of current split with the best entropy so far 
				if(best_entropy > average_entropy) {
					// the current split is better, keep it 
					best_entropy = average_entropy;
					best_iX = iX;
					best_threshold = threshold;
				}
			}
		}

		// split node if there is an improvement in entropy 
		if(best_entropy < node_entropy) {
			// there is an improvement in entropy, so we apply the split 
			// we are creating two extra nodes: treeSize and treeSize+1 
			current_node->attribute = best_iX;
			current_node->threshold = best_threshold;
			current_node->son1 = new Node;
			current_node->son1->number = treeSize;
			current_node->son2 = new Node;
			current_node->son2->number = treeSize+1;
			treeSize += 2;

			printf("  split node %d with question X[%d]<=%g (entropy improvement=%g) to create nodes %d and %d\n",current_node->number,best_iX,best_threshold,(node_entropy-best_entropy),current_node->son1->number,current_node->son2->number);

			// shift samples so that the first samples are X[best_iX] <=  best_threshold
			// and the last samples are X[best_iX] > best_threshold
			// (assume that all samples from first_sample to k-1 are X[best_iX] <=  best_threshold)
			k = first_sample;
			for(i=first_sample; i<=last_sample; i++) {
				if(X[best_iX][i] <=  best_threshold) {
					// move this sample to position k 
					if(i !=  k) {
						y = Y[k]; Y[k] = Y[i]; Y[i] = y;
						for(iX = 0; iX<nX; iX++) {
							x = X[iX][k]; X[iX][k] = X[iX][i]; X[iX][i] = x;
						}
					}
					// increase k by 1 
					k++;
				}
			}

			// now split recursively 
			Split_node(current_node->son1,depth+1,first_sample,k-1);
			Split_node(current_node->son2,depth+1,k,last_sample);
		}
	} else {
		// this is a leaf
	}
}  

/*************************************************************************/
// save_tree
// This function take save a tree in a file
void Save_tree(char * filename, Node *node) {
	static FILE *out;
	static vector<Node *> nodes;

	if(node == root_node) {
		// this is the first call, open the file
		if((out = fopen(filename,"w")) == NULL) {
			printf("*** Error opening file %s\n",filename);
			getchar();
			exit(1);
		}
		nodes.resize(treeSize);
	}

	nodes[node->number] = node; // keep node in right position

	if(node->son1) Save_tree(filename,node->son1);
	if(node->son2) Save_tree(filename,node->son2);

	if(node == root_node) {
		for(int i=0; i<treeSize; i++) {
			fprintf(out,"%d %d %d %g %d",nodes[i]->number,nodes[i]->depth,nodes[i]->attribute,nodes[i]->threshold,nodes[i]->category);
			if(nodes[i]->son1) fprintf(out," %d",nodes[i]->son1->number);
			if(nodes[i]->son2) fprintf(out," %d",nodes[i]->son2->number);
			fprintf(out,"\n");
		}
		fclose(out);
	}
}

/*************************************************************************/
// load_tree
// This function read a tree structure from a file
Node *Load_tree(char * filename) {
	FILE *in;
	vector<Node *> nodes;
	char line[10240];
	int number;
	int depth;
	int category;
	int attribute;
	float threshold;
	int son1;
	int son2;

	if((in = fopen(filename,"r")) == NULL) {
		printf("*** Error opening file %s\n",filename);
		getchar();
		exit(1);
	}

	// first read the file to compute the number of nodes in the tree 
	treeSize = 0;
	while(fgets(line,sizeof(line),in)) {
		treeSize++;
	}

	rewind(in);
	nodes.resize(treeSize);
	for(int i=0; i<treeSize; i++) nodes[i] = new Node;

	// read the tree from the file and fill in the tree structure 
	while(fscanf(in,"%d %d %d %g %d",&number, &depth, &attribute, &threshold, &category) == 5) {
		printf("  node %d depth=%d attribute=%d threshold=%g category=%d",number,depth,attribute,threshold,category);
		nodes[number]->number = number;
		nodes[number]->depth = depth;
		nodes[number]->attribute = attribute;
		nodes[number]->threshold = threshold;
		nodes[number]->category = category;
		if(attribute >= 0) {
			fscanf(in,"%d %d",&son1, &son2);
			nodes[number]->son1 = nodes[son1];
			nodes[number]->son2 = nodes[son2];
			printf(" son1=%d son2=%d\n",son1,son2);
		} else {
			nodes[number]->son1 = NULL;
			nodes[number]->son2 = NULL;
			printf("\n");
		}
		nodes[number]->falseNb = 0;
		nodes[number]->totalNb = 0;
		nodes[number]->entropy = 0;
	}
	fclose(in);
	printf("Tree with %d nodes read from file %s\n",treeSize,filename);
	return nodes[0];
}

void evalSample(Node *node, int i) {
	if(node->category != Y[i]) {
		node->falseNb += 1;
	}
	/*if(node->number == 0) {
		cout << "Node 0 category: " << node->category << ", sample: " << Y[i] << endl;
	}*/
	node->totalNb += 1;
}

/**************************************************************************/
// computes the entropy for each node (==> ONLY IN PREDICTION MODE <==) and prints entropy and error rate
float computeEntropy(Node *node) {
	float falseRate;
	int n1;
	int n2;

	if(node->totalNb) {
		falseRate = node->falseNb/node->totalNb;
	}
	else {
		falseRate = 0;
	}

	if(node->son1) {
		n1 = node->son1->totalNb;
		n2 = node->son2->totalNb;
		node->entropy = (computeEntropy(node->son1)*n1 + computeEntropy(node->son2)*n2)/(n1 + n2);
	}
	else {
		if(falseRate > 0) {
			node->entropy -= falseRate*log(falseRate)/Log2;
		}
		if(falseRate < 1) {
			node->entropy -= (1-falseRate)*log(1-falseRate)/Log2;
		}
	}
	cout << node->number << "\t" << node->entropy << "\t" << falseRate << "\t" << node->falseNb << endl;
	return node->entropy;
}

/*************************************************************************/
// predicts the category of one data sample using a tree
int Predict(Node *node, int sample) {

	evalSample(node, sample);

	/******************************************************************/
	/* BEGIN IMPLEMENTATION                                           */
	/******************************************************************/
	// if node is final, just return node category
	if (node->son1 && node->son2) {
		if (X[node->attribute][sample] <= node->threshold)
		{
			Predict(node->son1, sample);
		}
		else {
			Predict(node->son2, sample);
			}
		
	}
	else
	{
		return node->category;
	}
	/* END IMPLEMENTATION                                             */
	/******************************************************************/
	return 0;
}

/*************************************************************************/
// predicts the training data using a tree
// for each node display the entropy and error
void Prediction(Node *node) {
	int n0;
	int n1;
	int i;
	int y;
	int nok;
	int ncorrectlydetected;
	int nmissed;
	int nfalsealarm;

	nok = ncorrectlydetected = nmissed = nfalsealarm = 0;

	for(i=0; i<nT; i++) {
		y = Predict(node,i);
		if(Y[i] == 1)
			if(y == 1) ncorrectlydetected++;
			else nmissed++;
		else if(y == 0) nok++;
			else nfalsealarm++;
	}

	printf("Test data: %d samples, %d errors\n",nT,nmissed+nfalsealarm);
	printf("  %d ok\n",nok);
	printf("  %d intrusions correctly detected\n",ncorrectlydetected);
	printf("  %d intrusions missed\n",nmissed);
	printf("  %d false alarms\n",nfalsealarm);
}

/*************************************************************************/
// this is the main procedure
int main(int argc, char **argv) {

	if(argc != 4) {
		printf("Syntax: <program> -t data_filename tree_filename\n");
		printf("\t training: grow a tree file the data\n");
		printf("Syntax: <program> -p data_filename tree_filename\n");
		printf("\t prediction: predict the data with a tree\n");
		printf("\t\t data_filename: contains the data\n");
		printf("\t\t tree_filename: file containing the tree\n");
		getchar();
		exit(1);
	}

	if (strcmp(argv[1],"-t") == 0) {
		printf("Running in training mode.\n");

		// read file containing training data 
		Read_data(argv[2],"0");
		printf("Data set: %d samples, %d attributes\n",nT,nX);

		// start processing 
		max_depth = 4; // you may change the value of max_depth according to your requirement

		// recursively split node to construct the tree 
		// the initial node contains all the data 
		root_node = new Node;
		root_node->number = 0;
		treeSize = 1;

		Split_node(root_node,0,0,nT-1);
		printf("The complete tree has %d nodes (non-splitting nodes are included)\n",treeSize);

		Save_tree(argv[3],root_node);
		printf("Tree (%d nodes) saved in: %s\n",treeSize,argv[3]);
	} else if(strcmp(argv[1],"-p") == 0) {
		printf("Running in prediction mode.\n");
		// read file containing training data 
		Read_data(argv[2],"0");
		printf("Data set: %d samples, %d attributes\n",nT,nX);

		// start processing 
		root_node = Load_tree(argv[3]);
		Prediction(root_node);
		cout << "node\tentropy\terror rate\tnb of errors" << endl;
		computeEntropy(root_node);
	} else {
		printf("mode %s is unknown, exiting\n",argv[1]);
		exit(1);
	}

	printf("Finished, press Enter to exit.\n");
	getchar();
	return 1;
}
/*************************************************************************/
