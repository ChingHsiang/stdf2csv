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
#define maxrows 	12000000
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

map<string, string> rec_data::ProcessPTR()
{
	map<string, string> ptr;

	int testnum= (unsigned char) data[0] | (unsigned char) data[1] << 8 | (unsigned char) data[2] << 16 | (unsigned char) data[3] << 24;

	ptr["test_num"] =  to_string(testnum);

	char head_num = data[4];  ptr["head_num"] = to_string(head_num);
	char site_num = data[5];  ptr["site_num"] = to_string(site_num);
	unsigned char test_flg = data[6];  ptr["test_flg"] = to_string(test_flg);
	unsigned char param_flg = data[7]; ptr["param_flg"] = to_string(param_flg);

	string results = "";
	string tst_flg = "";
	string lolimit = "";
	string uplimit = "";

	if ((test_flg & 31) == 0 & (param_flg & 7) == 0) // if zero, result is not empty
	{
		float f;
		unsigned char b[] = { data[8], data[9], data[10], data[11] };
		memcpy(&f, &b, sizeof(f));
		results = to_string(f);
	}

	ptr["results"] = results;

	if ((test_flg & (1 << 5)) != 0 | (test_flg & (1 << 4)) != 0 | (test_flg & (1 << 3)) != 0) tst_flg = "N";

	if ((test_flg & (1 << 7)) != 0 & (test_flg & (1 << 6)) == 0) tst_flg = "F";

	if ((test_flg & (1 << 6)) != 0) tst_flg = "X";  //no pass fail

	if ((test_flg & (1 << 0)) != 0) tst_flg = "A";  //alarm

	ptr["tst_flg"] = tst_flg;

	int pt = 12, len = data[pt];

	char* t_testnam = new char[len];

	memcpy(t_testnam, data + pt + 1, len);
	t_testnam[len] = 0;

	string test_nam = t_testnam; ptr["test_nam"] = test_nam;

	pt = pt + len + 1;


	if (pt < _len)
	{
		len = data[pt];
		char* t_alarm_id = new char[len];
		memcpy(t_alarm_id, data + pt +1 , len);
		t_alarm_id[len] = 0;
		string alarm_id = t_alarm_id;//Encoding.ASCII.GetString(data.Skip(pt + 1).Take(len).ToArray<byte>());
		ptr["alarm_id"] = alarm_id;
		pt = pt + len + 1;
		unsigned char opt_flg = data[pt]; ptr["opt_flg"] = opt_flg;


		char res_scal = data[++pt];
		char llm_scal = data[++pt];
		char hlm_scal = data[++pt];

		//rewrite opt flag

		if ((opt_flg & (1 << 4)) == 0 & (opt_flg & (1 << 6)) == 0)  //np then transform limits
		{
			float ll;
			unsigned char b[] = { data[pt + 1], data[pt + 2], data[pt + 3], data[pt + 4] };
			memcpy(&ll, &b, sizeof(ll));
			lolimit = to_string(ll);
		}
		if ((opt_flg & (1 << 5)) == 0 & (opt_flg & (1 << 7)) == 0)
		{
			float uu;
			unsigned char b[] = { data[pt + 5], data[pt + 6], data[pt + 7], data[pt + 8] };
			memcpy(&uu, &b, sizeof(uu));
			uplimit = to_string(uu);
		}


		pt = pt + 9;

		if (pt < _len)
		{
			len = data[pt];
			char* t_units = new char[len];
			memcpy(t_units, data + pt +1 , len);
			t_units[len] = 0;
			string units = t_units; ptr["units"] = units;

			pt = pt + len + 1; len = data[pt];
			char* t_resfmt = new char[len];
			memcpy(t_resfmt, data + pt+1 , len);
			t_resfmt[len] = 0;
			string c_resfmt = t_resfmt;//Encoding.ASCII.GetString(data.Skip(pt + 1).Take(len).ToArray<byte>());
			ptr["c_resfmt"] = c_resfmt;

			pt = pt + len + 1; len = data[pt];
			char* t_llmfmt = new char[len];
			memcpy(t_llmfmt, data + pt+ 1, len);
			t_llmfmt[len] = 0;
			string c_llmfmt = t_llmfmt;//Encoding.ASCII.GetString(data.Skip(pt + 1).Take(len).ToArray<byte>());
			ptr["c_llmfmt"] = c_llmfmt;

			pt = pt + len + 1; len = data[pt];
			char* t_hlmfmt = new char[len];
			memcpy(t_hlmfmt, data + pt +1, len);
			t_hlmfmt[len] = 0;
			string c_hlmfmt = t_hlmfmt;//Encoding.ASCII.GetString(data.Skip(pt + 1).Take(len).ToArray<byte>());
			ptr["c_hlmfmt"] = c_hlmfmt;
			pt = pt + len + 1;

			if (pt < _len)
			{
				if ((opt_flg & (1 << 2)) == 0)
				{
					float ll_s;
					unsigned char b[] = { data[pt], data[pt + 1], data[pt + 2], data[pt + 3] };
					memcpy(&ll_s, &b, sizeof(ll_s));
					string lolimit_s = to_string(ll_s); ptr["lolimit_s"] = lolimit_s;
				}
				if ((opt_flg & (1 << 3)) == 0)
				{
					float uu_s;
					unsigned char b[] = { data[pt+4], data[pt + 5], data[pt + 6], data[pt + 7] };
					memcpy(&uu_s, &b, sizeof(uu_s));
					string uplimit_s = to_string(uu_s); ptr["uplimit_s"] = uplimit_s;
				}
			}
		}


	}

	ptr["lolimit"] = lolimit;
	ptr["uplimit"] = uplimit;

	return ptr;
}

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

__device__ int strlen2(const char *s)
{
	int i = 0;
	while(s[i] != '\0')
	{
		i++;
	}
	return i;
}
__device__ char* cstrcat2(char *dest,char *src,int to_len,int *sp)
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
/*
__device__ char* cstrcat3(char *dest,unsigned char *src,int *sp)
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

	//vector<char> ret;
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
		//resultbuf={data[8],data[9],data[10],data[11]};
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
			//uu={data[pt + 5],data[pt + 6],data[pt + 7],data[pt + 8]};

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
	if(testnum>0)
	{
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

}
*/
__device__ void PTRdecoder(char* data, int *_len)
{
	//vector<char> ret;
		char outstr[maxcols]={'\0'};
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
			//resultbuf={data[8],data[9],data[10],data[11]};
		}

		if ((test_flg & (1 << 5)) != 0 | (test_flg & (1 << 4)) != 0 | (test_flg & (1 << 3)) != 0) tst_flg = 'N';

		if ((test_flg & (1 << 7)) != 0 & (test_flg & (1 << 6)) == 0) tst_flg = 'F';

		if ((test_flg & (1 << 6)) != 0) tst_flg = 'X';  //no pass fail

		if ((test_flg & (1 << 0)) != 0) tst_flg = 'A';  //alarm


		int pt = 12, len = data[pt], tnamlen=len;

		char *testnam = { '\0' };

		testnam = new char[len];

		for(int i =0; i<len;i++)
		{
			testnam[i]=data[pt+1+i];
		}

		pt = pt + len + 1;


		if (pt < *_len)
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
				//uu={data[pt + 5],data[pt + 6],data[pt + 7],data[pt + 8]};

			}

			pt = pt + 9;

			if (pt < *_len)
			{
				len = data[pt];

				for(int i =0; i<len;i++)
				{
					unit[i]=data[pt+1+i];
				}

			}
		}

		int sp=0;

		cstrcat2(outstr,data,4,&sp);
		cstrcat2(outstr,testnam,tnamlen,&sp);
		cstrcat2(outstr,unit,strlen2(unit),&sp); //3

		cstrcat2(outstr,(char *)ll,4,&sp); //4
		cstrcat2(outstr,(char *)uu,4,&sp);
		cstrcat2(outstr,(char *)resultbuf,4,&sp);

		outstr[sp]=tst_flg;   outstr[++sp]='|';
		outstr[++sp]=head_num;outstr[++sp]='|';
		outstr[++sp]=site_num;outstr[++sp]='|';


		memcpy(data,outstr,sp);

		*_len=sp;

		/*
		sprintf(outstr,"%d|",testnum);
		int sp=strlen2(outstr);
		if(testnum>0)
		{
			cstrcat2(outstr,testnam,tnamlen,&sp);
			cstrcat2(outstr,unit,strlen2(unit),&sp); //3

			cstrcat3(outstr,ll,&sp); //4
			cstrcat3(outstr,uu,&sp);
			cstrcat3(outstr,resultbuf,&sp);

			outstr[sp]=tst_flg;   outstr[++sp]='|';
			outstr[++sp]=head_num;outstr[++sp]='|';
			outstr[++sp]=site_num;outstr[++sp]='|';
		}
		*/


}
//vector<tempObject>().swap(tempVector); use this to clear memory

__device__ void RunTest(int *d)
{
	*d=*d+1;
}
__global__ void Decode(int *size,char * data) {
	/**********************************************************************
	*     Initialize points on line
	*********************************************************************/
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//int row = blockDim.y * blockIdx.y + threadIdx.y ;
	//int col = blockDim.x * blockIdx.x + threadIdx.x ;
	//printf("values in idx: %d", idx);

	//int p=size[idx];
	//RunTest(&size[idx]);

	//printf("idx: %d len: %d mod:%d pitch: %d\n", idx, p,size[idx]);

	PTRdecoder(data+idx*150,&size[idx]);


	/*
	for(int i=0;i<size[idx];i++)
	{
		printf(" values %c",data[i]);
	}
	printf("\n");

	*/
	//printf("values in idx: %d is %d\n", idx, size[idx]);


	//float x, fac;
	//float nowval,	/* values at time t */
	//	oldval,		/* values at time (t-dt) */
	//	newval;		/* values at time (t+dt) */
	/*
	 Calculate initial values based on sine curve */
	//fac = 2.0 * PI;
	//x = (-1.0 + idx) / (float) (tpoints - 1);

	//nowval = sin(fac * x);

	/* Initialize old values array */
	//oldval = nowval;

}
__global__ void Decode2(int *size,char * data,size_t pitch) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	PTRdecoder(data+idx*pitch,&size[idx]);

}
__host__ string float2str(char * data,int cloc,int tloc)
{

	float f;
	unsigned char b[] = { (unsigned char) data[cloc+tloc+1], (unsigned char)data[cloc+tloc+2], (unsigned char) data[cloc+tloc+3], (unsigned char)data[cloc+tloc+4]};
	memcpy(&f, &b, sizeof(f));
	string str = "";
	str=to_string(f);

	return str;
}
__host__ void GPU2OUT(int* arr,char* ptr_arr, string outpath,int size)
{
		size_t pitch=128;
		//int* arr= &arrlen[0];

		int* sizedata;
		char* gpudata;

		//cout<<sizeof(arr[s]) <<endl;

		int blockCount=size /1024 +1 ;
		int arySize= (blockCount*1024+1)*sizeof(int);
		cudaFn(cudaMalloc, &sizedata, arySize);
		//cudaFn(cudaMalloc, &gpudata, (arrlen.size()+10)*maxcols*sizeof(char));
		//cudaFn(cudaMallocPitch, &gpudata, &pitch, maxcols, size*sizeof(char));
		cudaFn(cudaMallocPitch, &gpudata, &pitch, maxcols, size*sizeof(char));


		//memcpy
		cudaFn(cudaMemcpy, sizedata, arr, arySize, cudaMemcpyHostToDevice);
		//cudaFn(cudaMemcpy, gpudata, ptr_arr, (arrlen.size()+10)*maxcols*sizeof(char), cudaMemcpyHostToDevice);
		cudaFn(cudaMemcpy2D, gpudata, pitch, ptr_arr, maxcols, maxcols, size
				*sizeof(char),cudaMemcpyHostToDevice);


		//kernelFn(Decode, blockCount, 1024, sizedata,gpudata);
		kernelFn(Decode2, blockCount, 1024, sizedata, gpudata, pitch);


		//cudaFn(cudaMemcpy, ptr_arr, gpudata, (arrlen.size()+10)*maxcols*sizeof(char), cudaMemcpyDeviceToHost);
		cudaFn(cudaMemcpy2D, ptr_arr, maxcols, gpudata, pitch, maxcols, size*sizeof(char),cudaMemcpyDeviceToHost);
		cudaFn(cudaMemcpy, arr,sizedata, arySize, cudaMemcpyDeviceToHost);

		cudaFn(cudaFree, gpudata);
		cudaFn(cudaFree, sizedata);
		cudaFn(cudaDeviceReset);

		//output data

		fstream ofile;
		ofile.open(outpath,ios::out | ios::app);

		for(int i=0;i<size;i++)
		{
			//cout<< arr[i]<<endl;
			int dcnt=0, tloc=0;
			int cloc=i*maxcols;
			//ofile << arr[i] << ',';
			int testnum= (unsigned char) ptr_arr[cloc+0] | (unsigned char) ptr_arr[cloc+1] << 8 | (unsigned char) ptr_arr[cloc+2] << 16 | (unsigned char) ptr_arr[cloc+3] << 24;
			ofile <<testnum;
			for (int j = 4;j<arr[i];j++)
			{

				if(ptr_arr[cloc+j]=='|')
				{
					dcnt++;
					if(dcnt==3)
					{
						tloc=j;
						ofile<< '|' << float2str(ptr_arr,cloc, tloc) << '|'<<float2str(ptr_arr,cloc,tloc+5) << '|' <<float2str(ptr_arr,cloc,tloc+10);
						j=tloc+15;
					}
				}

				ofile << ptr_arr[cloc+j];
			}


			ofile<<endl;
		}

		ofile.close();


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

		//cout <<stdfpath <<endl;
		//string stdfpath = "/local2/mnt/workspace/cchsian/script/stdf2csv/dataset/ASE_ASEK-S_TD90-PJ928-1_QL8U-PJ928-1-02_RC_5_000FE9144SW.0E00POUNDT6H286.00_2019-05-02T133908_2_ENF1_946QM421.stdf";
			//const char *stdfpath = "C:\\Users\\cchsian\\Desktop\\stdf\\demo\\ASE_ASEK-S_CD90-NN875-6_QAFU-NN875-6-5_000NZ750QQ8.0E051-GR81107303A2_2018-05-29T223356_2_PSQA1_87NQM32111.stdf";

		size_t lastindex = stdfpath.find_last_of(".");
		string rawname = stdfpath.substr(0, lastindex)+string("_cuda.csv");

		ifstream stdf;

		stdf.open(stdfpath, ios::binary);

		Dic dic = RecCode();
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

							//char *dt = new char[len];
							//stdf.read(dt, len * sizeof(char));
							//rec_data rt(dt, len);
							//arrlen.push_back(len);
							//ptrdata.push_back(rt); //get ptr data

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


			}
			//cout<<arrlen.size() <<endl;
			arr= &arrlen[0];
			GPU2OUT(arr,ptr_arr,rawname,arrlen.size());
			vector<int>().swap(arrlen);


			/*

			size_t pitch;
			int* arr= &arrlen[0];

			int* sizedata;
			char* gpudata;

			int blockCount=arrlen.size() /1024 +1 ;
			int arySize= (blockCount*1024+1)*sizeof(int);
			cudaFn(cudaMalloc, &sizedata, arySize);
			//cudaFn(cudaMalloc, &gpudata, (arrlen.size()+10)*maxcols*sizeof(char));
			cudaFn(cudaMallocPitch, &gpudata, &pitch, maxcols, arrlen.size()*sizeof(char));

			//memcpy
			cudaFn(cudaMemcpy, sizedata, arr, arySize, cudaMemcpyHostToDevice);
			//cudaFn(cudaMemcpy, gpudata, ptr_arr, (arrlen.size()+10)*maxcols*sizeof(char), cudaMemcpyHostToDevice);
			cudaFn(cudaMemcpy2D, gpudata, pitch, ptr_arr, maxcols, maxcols, arrlen.size()*sizeof(char),cudaMemcpyHostToDevice);


			//kernelFn(Decode, blockCount, 1024, sizedata,gpudata);
			kernelFn(Decode2, blockCount, 1024, sizedata,gpudata,pitch);

			cudaFn(cudaMemcpy, arr,sizedata, arySize, cudaMemcpyDeviceToHost);
			//cudaFn(cudaMemcpy, ptr_arr, gpudata, (arrlen.size()+10)*maxcols*sizeof(char), cudaMemcpyDeviceToHost);
			cudaFn(cudaMemcpy2D, ptr_arr, maxcols, gpudata, pitch, maxcols, arrlen.size()*sizeof(char),cudaMemcpyDeviceToHost);

			cudaFn(cudaFree, gpudata);
			cudaFn(cudaFree, sizedata);
			cudaFn(cudaDeviceReset);

			//output data

			fstream ofile;
			ofile.open(rawname,ios::out | ios::app);

			for(int i=0;i<arrlen.size();i++)
			{
				int testnum= (unsigned char) ptr_arr[i*maxcols+0] | (unsigned char) ptr_arr[i*maxcols+1] << 8 | (unsigned char) ptr_arr[i*maxcols+2] << 16 | (unsigned char) ptr_arr[i*maxcols+3] << 24;
				ofile <<testnum;
				for (int j = 4;j<arr[i];j++)
				{
					ofile << ptr_arr[i*maxcols+j];
				}
				ofile<<endl;
			}

			ofile.close();

			*/

			//vector<int>().swap(arrlen);
			//delete[] ptr_arr;
			//ptr_arr=new char[maxrows];




			/*
			fstream ofile;
			ofile.open("/usr2/cchsian/Desktop/stdf/serial_bug.csv",ios::out);
			ofile << "test_num,test_name,units,l_limit,u_limit,results,passfail,headnum,sitenum" << endl;
			for (int i = 0;i < rcnt;++i)
			{

				ProcessPTR(ptr_arr + i * 150, arrlen[i], ofile);
							}
			ofile.close();
			*/

		}


		stdf.close();

	}

	return 0;
}
