/* setting */

#define SAVE_IMAGE

/* headers */
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>

extern "C" {
	#include <lua.h>
	#include <lualib.h>
	#include <lauxlib.h>
}


/* namespace */
using namespace std;
using namespace cv;


/* short func */
#define pow2(x) ((x)*(x))
#define dis(A, B) sqrt(pow2(A.x - B.x) + pow2(A.y - B.y))



/* const var */
const int MAXN = 100;
const int THRESHOLD = 10000;
const Size BLUR_AREA = Size(50, 50);


/* global var */
#ifdef SAVE_IMAGE
Mat ori_im;
#endif
FILE * f_in, * f_out;


void label(char * input_path, char * type) {
	// load
	Mat im = imread(input_path, CV_LOAD_IMAGE_GRAYSCALE);

	// preparation
	blur(im, im, BLUR_AREA);
	threshold(im, im, 0, 255, CV_THRESH_OTSU);

	// find contours
	vector< vector< Point > > contours;
	vector< Vec4i > hierarchy;
	findContours(im, contours, hierarchy, RETR_EXTERNAL,CHAIN_APPROX_NONE, Point());


	// save rect points
	Point2f P[4];
	int n = contours.size();
	for (int i = 0, minx, miny, maxx, maxy; i < n; i ++) {
		RotatedRect rect = minAreaRect(contours[i]);
		rect.points(P);

		if (dis(P[0], P[1]) * dis(P[1], P[2]) < THRESHOLD)
			continue;

		minx = max((int)min(P[0].x, P[2].x), 0);
		maxx = (int)max(P[0].x, P[2].x);
		miny = max((int)min(P[0].y, P[2].y), 0);
		maxy = (int)max(P[0].y, P[2].y);

		fprintf(f_out, "%s %d %d %d %d\n", type, minx, maxx, miny, maxy);

#ifdef SAVE_IMAGE
		drawContours(ori_im, contours, i, Scalar(255), 1, 8, hierarchy);
		line(ori_im, Point(minx, miny), Point(maxx, miny), Scalar(255), 2);
		line(ori_im, Point(maxx, maxy), Point(minx, maxy), Scalar(255), 2);
		line(ori_im, Point(maxx, maxy), Point(maxx, miny), Scalar(255), 2);
		line(ori_im, Point(maxx, maxy), Point(minx, maxy), Scalar(255), 2);
		putText(ori_im, type, P[1], FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 3, 3);
#endif
	}

}


extern "C" int l_label(lua_State * L) {
	f_in = fopen("/data8T/aucid/guideDogBackend/fastSceneUnderstanding/list.txt", "r");
	f_out = fopen("/data8T/aucid/guideDogBackend/fastSceneUnderstanding/points.txt", "w");

	int n;
	fscanf(f_in, "%d", &n);

	char s1[MAXN], s2[MAXN], dest[MAXN];

#ifdef SAVE_IMAGE
	fscanf(f_in, "%s", dest);

	// load original im
	ori_im = imread(dest);
	if (ori_im.empty()) {
		cout << dest << " is empty!\n";
		return 0;
	}
#endif

	for (int i = 0; i < n; i ++) {
		fscanf(f_in, "%s", s1);
		fscanf(f_in, "%s", s2);

		cout << "labeling " << s2 << endl;
		label(s1, s2);
	}

#ifdef SAVE_IMAGE
	imwrite(dest, ori_im);
#endif

	fclose(f_in);
	fclose(f_out);

	return 0;
}


static luaL_reg liblabel[] = {
	{ "l_label" , l_label },
	{ NULL, NULL }
};


extern "C" int luaopen_liblabel(lua_State * L){
	luaL_register(L, "liblabel", liblabel); // lua 5.1

	return 1;
}
