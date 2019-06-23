// stdf2csvpth.cpp : Defines the entry point for the console application.
//


#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <math.h>
#include <time.h>
#include <string.h>
#include <errno.h>
#include <numeric>
#include <map>
#include <vector>



#define maxrows 4000000000
#define maxcols 150

#define n_threads 8
#define num_thread 16


using namespace std;
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

typedef map<pair<char, char>, type_t> mMap;

typedef struct infostrct
{
	int thid;
	int start;
	int numrows;
} info;


mMap RecType() {

	mMap mp;
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
	mp[make_pair(0, 10)] = FAR;

	return mp;
};

char *ptr_arr = new char[maxrows];
int *lenarr;

char* cstrcat2(char *dest, char *src, int to_len, int *sp)
{
	int i = *sp;
	int j = 0;

	while (j<to_len)
	{
		dest[i] = src[j];
		j++;
		i++;
	}
	dest[i] = '|';
	*sp = i + 1;
	return dest;
}
char* cstrcat3(char *dest, unsigned char *src, int *sp)
{

	int i = *sp, c = 0;

	float f;
	unsigned char b[] = { src[0], src[1], src[2], src[3] };

	memcpy(&f, &b, sizeof(f));

	char buff[20];
	int ret = snprintf(buff, sizeof(buff), "%f|", f);

	if (ret<0)
	{
		dest[i] = '|';
		*sp = i + 1;
	}
	else
	{
		int len = strlen(buff);
		while (c<len)
		{
			dest[i] = buff[c];
			c++;i++;
		}
		*sp = i;

	}
	return dest;
}

void ProcessPTR(char* data, int *_len)
{

	//vector<char> ret;
	char outstr[150] = { '\0' };
	int testnum = (unsigned char)data[0] | (unsigned char)data[1] << 8 | (unsigned char)data[2] << 16 | (unsigned char)data[3] << 24;

	//char special_char = {'|'};
	unsigned char head_num = (int)data[4] + '0';
	unsigned char site_num = (int)data[5] + '0';


	unsigned char resultbuf[4] = { '\0' };
	unsigned char ll[4] = { '\0' };
	unsigned char uu[4] = { '\0' };
	char unit[10] = { '\0' };
	char tst_flg = '\0';

	unsigned char test_flg = data[6];
	unsigned char param_flg = data[7];


	if ((test_flg & 31) == 0 & (param_flg & 7) == 0) // if zero, result is not empty
	{

		resultbuf[0] = data[8];
		resultbuf[1] = data[9];
		resultbuf[2] = data[10];
		resultbuf[3] = data[11];
		//resultbuf={data[8],data[9],data[10],data[11]};
	}

	if ((test_flg & (1 << 5)) != 0 | (test_flg & (1 << 4)) != 0 | (test_flg & (1 << 3)) != 0) tst_flg = 'N';

	if ((test_flg & (1 << 7)) != 0 & (test_flg & (1 << 6)) == 0) tst_flg = 'F';

	if ((test_flg & (1 << 6)) != 0) tst_flg = 'X';  //no pass fail

	if ((test_flg & (1 << 0)) != 0) tst_flg = 'A';  //alarm


	int pt = 12, len = data[pt], tnamlen = len;

	char *testnam = { '\0' };

	testnam = new char[len];

	for (int i = 0; i<len;i++)
	{
		testnam[i] = data[pt + 1 + i];
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

			ll[0] = data[pt + 1];
			ll[1] = data[pt + 2];
			ll[2] = data[pt + 3];
			ll[3] = data[pt + 4];
			//ll={data[pt + 1],data[pt + 2],data[pt + 3],data[pt + 4]};

		}
		if ((opt_flg & (1 << 5)) == 0 & (opt_flg & (1 << 7)) == 0)
		{

			uu[0] = data[pt + 5];
			uu[1] = data[pt + 6];
			uu[2] = data[pt + 7];
			uu[3] = data[pt + 8];
			//uu={data[pt + 5],data[pt + 6],data[pt + 7],data[pt + 8]};

		}

		pt = pt + 9;

		if (pt < *_len)
		{
			len = data[pt];

			for (int i = 0; i<len;i++)
			{
				unit[i] = data[pt + 1 + i];
			}

		}
	}

	sprintf(outstr, "%d|", testnum);
	int sp = strlen(outstr);

	cstrcat2(outstr, testnam, tnamlen, &sp);
	cstrcat2(outstr, unit, strlen(unit), &sp); //3

	cstrcat3(outstr, ll, &sp); //4
	cstrcat3(outstr, uu, &sp);
	cstrcat3(outstr, resultbuf, &sp);
	outstr[sp] = tst_flg;   outstr[++sp] = '|';
	outstr[++sp] = head_num;outstr[++sp] = '|';
	outstr[++sp] = site_num;outstr[++sp] = '|';

	memcpy(data, outstr, sp);

	*_len = sp;
	
}

void* ptr_decoder(void* infoarr)
{
	info *arr;
	arr = (info *) infoarr;

	int d = 1;
	
	//printf("thread id: %d, rows: %d \n", arr->thid, arr->numrows);

	int loc = arr->start;

	for (int i = 0;i < arr->numrows;i++)
	{
		ProcessPTR(ptr_arr + (loc + i)*maxcols, lenarr + (loc + i));
	}
	return (void*)d;
}

int main(int arg, char *args[])
{
	if(arg<2)
	{
		return 0;
	}
	else
	{
		string stdfpath = args[1]; //"C:\\Users\\cchsian\\Desktop\\stdf\\demo\\ASE_ASEK-S_CD90-NN875-6_QAFU-NN875-6-5_000NZ750QQ8.0E051-GR81107303A2_2018-05-29T223356_2_PSQA1_87NQM32111.stdf";

		size_t lastindex = stdfpath.find_last_of(".");
		string rawname = stdfpath.substr(0, lastindex) + string("_pthrd16.csv");

		ifstream stdf;
		stdf.open(stdfpath, ios::binary);

		mMap mp = RecType();
		vector<int> arrlen;

		if (stdf)
		{
			//get file size
			long begin = stdf.tellg();
			stdf.seekg(0, ios::end);
			long end = stdf.tellg();
			long length = (end - begin);
			stdf.seekg(0, ios::beg);

			int rcnt = 0;

			while (stdf.tellg() < length)
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
						stdf.seekg(len, ios::cur);
					}
					break;
					case MIR:
					{
						stdf.seekg(len, ios::cur);
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
						stdf.read((char*)(ptr_arr + rcnt* maxcols), len * sizeof(char));
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


				//string code(reinterpret_cast< char* >(arr), sizeof(arr));

				//printf("%d\n",code);
			}
			stdf.close();
		}

		lenarr = &arrlen[0];

		pthread_t threads[num_thread];

		int averows, extra, rows = 0, size = arrlen.size();

		averows = size / num_thread;
		extra = size % num_thread;

		int tn; void* ret;

		info *infoarry = new info[num_thread];

		int tot_in = 0;
		
		int prev = 0;

		for (tn = 0;tn < num_thread;tn++)
		{
			rows = tn < extra ? averows + 1 : averows;
			infoarry[tn].start = prev;
			prev = prev + rows;
			infoarry[tn].numrows = rows;
			infoarry[tn].thid = tn;

			pthread_create(&threads[tn], NULL, ptr_decoder, (void *)&infoarry[tn]);
		}

		for (tn = 0;tn < num_thread;tn++) {
			pthread_join(threads[tn], &ret);
		}

		fstream ofile;
		ofile.open(rawname, ios::out | ios::app);

		for (int i = 0;i < arrlen.size();i++)
		{
			for (int j = 0;j < lenarr[i];j++)
			{
				ofile << ptr_arr[i*maxcols + j];
			}


			ofile << endl;
		}
	}

	//system("Pause");
	return 0;
}


