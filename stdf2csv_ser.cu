/*
 ============================================================================
 Name        : stdf2csv.cu
 Author      : ChingHsiang Chan
 Version     :
 Copyright   : Qualcomm Incorporations
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <fstream>

#include <map>
#include <string>
#include <sstream>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;



#define cudaFn(fn,...) { \
	cudaError_t __cudaStatus = fn(__VA_ARGS__); \
	if (__cudaStatus != cudaSuccess) { \
		fprintf(stderr, "Failed when calling cuda function \"%s\"!\n%s\n", #fn, cudaGetErrorString(__cudaStatus)); \
		exit(1); \
	} \
}
#define kernelFn(fn,block_count,thread_count,...) { \
	fn<<<block_count, thread_count>>>(__VA_ARGS__); \
	cudaError_t __cudaStatus = cudaGetLastError(); \
	if (__cudaStatus != cudaSuccess) { \
		fprintf(stderr, "%s launch failed: %s\n", #fn, cudaGetErrorString(__cudaStatus)); \
		exit(1); \
	} \
}

//#define maxrows 15000000
//#define maxrows   12000000
#define maxrows 	4000000000
#define maxcols 150
/*
static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

*
 * CUDA kernel that computes reciprocal values for a given vector

__global__ void reciprocalKernel(float *data, unsigned vectorSize) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx < vectorSize)
		data[idx] = 1.0/data[idx];
}
*/
/**
 * Host function that copies the data and launches the work on GPU

float *gpuReciprocal(float *data, unsigned size)
{
	float *rc = new float[size];
	float *gpuData;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuData, sizeof(float)*size));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuData, data, sizeof(float)*size, cudaMemcpyHostToDevice));
	
	static const int BLOCK_SIZE = 256;
	const int blockCount = (size+BLOCK_SIZE-1)/BLOCK_SIZE;
	reciprocalKernel<<<blockCount, BLOCK_SIZE>>> (gpuData, size);

	CUDA_CHECK_RETURN(cudaMemcpy(rc, gpuData, sizeof(float)*size, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaFree(gpuData));
	return rc;
}

float *cpuReciprocal(float *data, unsigned size)
{
	float *rc = new float[size];
	for (unsigned cnt = 0; cnt < size; ++cnt) rc[cnt] = 1.0/data[cnt];
	return rc;
}


void initialize(float *data, unsigned size)
{
	for (unsigned i = 0; i < size; ++i)
		data[i] = .5*(i+1);
}
 */

using namespace std;
/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.

static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}
*/
typedef enum
{
	FAR,
	ATR,
	MIR,
	MRR,
	PCR,
	HBR,
	SBR,
	PMR,
	PGR,
	PLR,
	RDR,
	SDR,
	WIR,
	WRR,
	WCR,
	PIR,
	PRR,
	TSR,
	PTR,
	MPR,
	FTR,
	BPS,
	EPS,
	GDR,
	DTR,
} type_t;

typedef map<pair<char, char>, string> Dic;
typedef map<pair<char, char>, type_t> mMap;


Dic RecCode() {

	Dic mp;

	mp[make_pair(0, 10)] = "FAR";

	mp[make_pair(0, 20)] = "ATR";

	mp[make_pair(1, 10)] = "MIR";
	mp[make_pair(1, 20)] = "MRR";
	mp[make_pair(1, 30)] = "PCR";
	mp[make_pair(1, 40)] = "HBR";
	mp[make_pair(1, 50)] = "SBR";
	mp[make_pair(1, 60)] = "PMR";
	mp[make_pair(1, 62)] = "PGR";
	mp[make_pair(1, 63)] = "PLR";
	mp[make_pair(1, 70)] = "RDR";
	mp[make_pair(1, 80)] = "SDR";

	mp[make_pair(2, 10)] = "WIR";
	mp[make_pair(2, 20)] = "WRR";
	mp[make_pair(2, 30)] = "WCR";

	mp[make_pair(5, 10)] = "PIR";
	mp[make_pair(5, 20)] = "PRR";

	mp[make_pair(10, 30)] = "TSR";

	mp[make_pair(15, 10)] = "PTR";
	mp[make_pair(15, 15)] = "MPR";
	mp[make_pair(15, 20)] = "FTR";
	mp[make_pair(20, 10)] = "BPS";
	mp[make_pair(20, 15)] = "EPS";

	mp[make_pair(50, 10)] = "GDR";
	mp[make_pair(50, 30)] = "DTR";

	return mp;
}
mMap RecType() {

	mMap mp;

	mp[make_pair(0, 10)] = FAR;

	mp[make_pair(0, 20)] = ATR;

	mp[make_pair(1, 10)] = MIR;
	mp[make_pair(1, 20)] = MRR;
	mp[make_pair(1, 30)] = PCR;
	mp[make_pair(1, 40)] = HBR;
	mp[make_pair(1, 50)] = SBR;
	mp[make_pair(1, 60)] = PMR;
	mp[make_pair(1, 62)] = PGR;
	mp[make_pair(1, 63)] = PLR;
	mp[make_pair(1, 70)] = RDR;
	mp[make_pair(1, 80)] = SDR;

	mp[make_pair(2, 10)] = WIR;
	mp[make_pair(2, 20)] = WRR;
	mp[make_pair(2, 30)] = WCR;

	mp[make_pair(5, 10)] = PIR;
	mp[make_pair(5, 20)] = PRR;

	mp[make_pair(10, 30)] = TSR;

	mp[make_pair(15, 10)] = PTR;
	mp[make_pair(15, 15)] = MPR;
	mp[make_pair(15, 20)] = FTR;
	mp[make_pair(20, 10)] = BPS;
	mp[make_pair(20, 15)] = EPS;

	mp[make_pair(50, 10)] = GDR;
	mp[make_pair(50, 30)] = DTR;

	return mp;
};

struct rec_data
{
	char *data;
	map<string, string> ProcessPTR();
	int _len;
	rec_data(char *arr,int Len)
	{
		data = arr;

		//int n =strlen(arr);
		_len = Len;
	}

};



void ccout(char s[])
{
	int c=0;

	while (s[c] != 0) {
	  printf("%c", s[c]);
	  c++;
	}
	printf("%c",'|');
}
char* cstrcpy(char *dest, char *src)
{
	int i=0;
	do {
		dest[i]=src[i];
	}while (src[i++]!=0);

	return dest;
}
char* cstrcat(char *dest,char *src)
{
	int i=0;
	while(dest[i]!=0) i++;
	cstrcpy(dest+i,src);
	return dest;
}
int cfindlast(char *dest)
{
	int size=99;
	while(dest[size]!='|' & size>=0)
	{
		size--;
	}
	return size;
}
void outputresults(char *dest,int *sp)
{
	for(int i=0;i<*sp;i++)
	{
		cout<< dest[i];
	}
	cout<<endl;
}

int strlen2(const char *s)
{
	int i = 0;
	while(s[i] != '\0')
	{
		i++;
	}
	return i;
}
char* cstrcat2(char *dest,char *src,int to_len,int *sp)
{
	int i=*sp;
	int j=0;

	while(j<to_len)
	{
		dest[i]=src[j];
		j++;
		i++;
	}
	dest[i]='|';
	*sp=i+1;
	return dest;
}

char* cstrcat3(char *dest,unsigned char *src,int *sp)
{

	int i=*sp,c=0;

	float f;
	unsigned char b[] = {src[0], src[1], src[2], src[3]};

	memcpy(&f, &b, sizeof(f));

	char buff[20];

	int ret=snprintf(buff,sizeof(buff),"%f|",f);

	if(ret<0)
	{
		dest[i]='|';
		*sp=i+1;
	}
	else
	{
		int len=strlen2(buff);
		while(c<len)
		{
			dest[i]=buff[c];
			c++;i++;
		}
		*sp=i;

	}
	return dest;

}
void ProcessPTR(char* data, int _len, fstream &ofile)
{

	char outstr[150]={'\0'};
	int testnum= (unsigned char) data[0] | (unsigned char) data[1] << 8 | (unsigned char) data[2] << 16 | (unsigned char) data[3] << 24;

	//char special_char = {'|'};
	unsigned char head_num = (int)data[4]+'0';
	unsigned char site_num = (int)data[5]+'0';


	unsigned char resultbuf[4]={'\0'};
	unsigned char ll[4]={'\0'};
	unsigned char uu[4]={'\0'};
	char unit[10]={'\0'};
	char tst_flg= '\0';

	unsigned char test_flg = data[6];
	unsigned char param_flg = data[7];


	if ((test_flg & 31) == 0 & (param_flg & 7) == 0) // if zero, result is not empty
	{

		resultbuf[0]=data[8];
		resultbuf[1]=data[9];
		resultbuf[2]=data[10];
		resultbuf[3]=data[11];
	}

	if ((test_flg & (1 << 5)) != 0 | (test_flg & (1 << 4)) != 0 | (test_flg & (1 << 3)) != 0) tst_flg = 'N';

	if ((test_flg & (1 << 7)) != 0 & (test_flg & (1 << 6)) == 0) tst_flg = 'F';

	if ((test_flg & (1 << 6)) != 0) tst_flg = 'X';  //no pass fail

	if ((test_flg & (1 << 0)) != 0) tst_flg = 'A';  //alarm


	int pt = 12, len = data[pt], tnamlen=len;

	char testnam[len] ={'\0'};

	for(int i =0; i<len;i++)
	{
		testnam[i]=data[pt+1+i];
	}

	pt = pt + len + 1;


	if (pt < _len)
	{
		len = data[pt];

		pt = pt + len + 1;

		unsigned char opt_flg = data[pt];

		char res_scal = data[++pt];
		char llm_scal = data[++pt];
		char hlm_scal = data[++pt];

		//rewrite opt flag

		if ((opt_flg & (1 << 4)) == 0 & (opt_flg & (1 << 6)) == 0)  //np then transform limits
		{

			ll[0]=data[pt + 1];
			ll[1]=data[pt + 2];
			ll[2]=data[pt + 3];
			ll[3]=data[pt + 4];
			//ll={data[pt + 1],data[pt + 2],data[pt + 3],data[pt + 4]};

		}
		if ((opt_flg & (1 << 5)) == 0 & (opt_flg & (1 << 7)) == 0)
		{

			uu[0]=data[pt + 5];
			uu[1]=data[pt + 6];
			uu[2]=data[pt + 7];
			uu[3]=data[pt + 8];


		}

		pt = pt + 9;

		if (pt < _len)
		{
			len = data[pt];

			for(int i =0; i<len;i++)
			{
				unit[i]=data[pt+1+i];
			}

		}
	}

	sprintf(outstr,"%d|",testnum);
	int sp=strlen(outstr);

	cstrcat2(outstr,testnam,tnamlen,&sp);
	cstrcat2(outstr,unit,strlen(unit),&sp); //3

	cstrcat3(outstr,ll,&sp); //4
	cstrcat3(outstr,uu,&sp);
	cstrcat3(outstr,resultbuf,&sp);
	outstr[sp]=tst_flg;   outstr[++sp]='|';
	outstr[++sp]=head_num;outstr[++sp]='|';
	outstr[++sp]=site_num;outstr[++sp]='|';

	for(int i=0;i<sp;i++)
	{
		ofile<< outstr[i];
	}

	ofile<<endl;
	

}


int main(int arg, char* args[])
{
	if(arg<2)
	{
		return 0;
	}
	else
	{
		string stdfpath= args[1];

		size_t lastindex = stdfpath.find_last_of(".");
		string rawname = stdfpath.substr(0, lastindex)+string("_ser.csv");

		ifstream stdf;

		stdf.open(stdfpath, ios::binary);

		//Dic dic = RecCode();
		mMap mp = RecType();

		vector<rec_data> mir;
		vector<int> arrlen;

		int cursize=0;

		char *ptr_arr = new char[maxrows];
		int* arr;

		if (stdf)
		{
			//get file size
			long begin = stdf.tellg();
			stdf.seekg(0, ios::end);
			long end = stdf.tellg();
			long length = (end - begin);
			stdf.seekg(0, ios::beg);
			int rcnt = 0;
			//read
			while (stdf.tellg()<length)
			{
				unsigned char head[4];

				stdf.read((char*)head, 4 * sizeof(char)); //rec

				int len = head[0] | head[1] << 8; //data length

				pair<char, char> code = make_pair(head[2], head[3]);

				if (mp.find(code) != mp.end())
				{
					switch (mp[code])
					{

						case FAR: //file start
						{
							//char *ptr;
							//ptr = (char*)malloc(len*sizeof(char));
							//char *dt = new char[len];
							//stdf.read(dt, len * sizeof(char));
							//rec_data rt(dt);
							//data.push_back(rt);
							//cout << strlen(dt) << endl;
							//cout << "FAR" << endl;
							//cout << stdf.tellg() << endl;
							stdf.seekg(len, ios::cur);
						}
						break;
						case MIR:
						{
							char *dt = new char[len];
							stdf.read(dt, len * sizeof(char));
							rec_data rt(dt, len);
							mir.push_back(rt);
							//data.push_back(rt);
							//cout << "MIR" << endl;
							//cout << stdf.tellg() << endl;
						}
						break;
						case WIR:
						{
							stdf.seekg(len, ios::cur);
						}
						break;
						case PIR:
						{
							stdf.seekg(len, ios::cur);

						}
						break;
						case PTR:
						{
							//ptr_arr[rcnt] = new char[maxcols];
							//stdf.read( (char *) (ptr_arr[rcnt]), len * sizeof(char));
							//rcnt++;

							stdf.read((char*) (ptr_arr +  rcnt* maxcols), len * sizeof(char));
							arrlen.push_back(len);
							rcnt++;


						}
						break;
						case PRR:
						{
							stdf.seekg(len, ios::cur);

						}
						break;
						case MRR: //last line
						{
							stdf.seekg(len, ios::cur);
						}
						break;
						default:
						{
							stdf.seekg(len, ios::cur);
						}
						break;
					}

				}
				else
				{
					stdf.seekg(len, ios::cur);
				}

				/*
				cursize=(rcnt*maxcols+10*maxcols)% maxrows;


				if(cursize==0)
				{
					//cout<<arrlen.size() <<endl;
					arr= &arrlen[0];

					GPU2OUT(arr,ptr_arr,rawname,arrlen.size());
					vector<int>().swap(arrlen);
					//delete[] ptr_arr;
					//ptr_arr=new char[maxrows];
					rcnt=0;

				}
				*/

			}

			fstream ofile;
			ofile.open(rawname,ios::out | ios::app);
			//ofile << "test_num,test_name,units,l_limit,u_limit,results,passfail,headnum,sitenum" << endl;
			for (int i = 0;i < rcnt;++i)
			{
				ProcessPTR(ptr_arr+i*maxcols,arrlen[i],ofile);

			}
			ofile.close();


		}


		stdf.close();

	}

	return 0;
}
