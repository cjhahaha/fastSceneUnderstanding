extern "C" {
	#include <lua.h>
	#include <lualib.h>
	#include <lauxlib.h>
}

#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define DEBUG
#define pow2(x) ((x)*(x))
#define dis(A, B) sqrt(pow2(A.x - B.x) + pow2(A.y - B.y))

#define THRESHOLD 1000
#define BLUR_AREA Size(30,30)


void label(char * input_path, char * type, char * ori_path, char * output_path) {
	// load
	Mat im = imread(input_path, CV_LOAD_IMAGE_GRAYSCALE);

	// preparation
	blur(im, im, BLUR_AREA);
	threshold(im, im, 0, 255, CV_THRESH_OTSU);

	// find contours
	vector< vector< Point > > contours;
	vector< Vec4i > hierarchy;
	findContours(im, contours, hierarchy, RETR_EXTERNAL,CHAIN_APPROX_NONE, Point());

#ifdef DEBUG
	// load original im
	Mat ori_im = imread(ori_path);
#endif

	// save rect points
	Point2f P[4];
	int n = contours.size();
	for (int i = 0; i < n; i ++) {
		RotatedRect rect = minAreaRect(contours[i]);
		rect.points(P);

		if (dis(P[0], P[1]) * dis(P[1], P[2]) < THRESHOLD)
			continue;

#ifdef DEBUG
		drawContours(ori_im, contours, i, Scalar(255), 1, 8, hierarchy);
		for (int j = 0; j < 4; j ++)
			line(ori_im,P[j], P[(j + 1) % 4], Scalar(255), 2);
		putText(ori_im, type, P[1], FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 3, 3);
#endif
	}

#ifdef DEBUG
	imwrite(output_path, ori_im);
#endif
}


extern "C" int l_label(lua_State * L) {
	string input_path = lua_tostring(L, 1);
	string type = lua_tostring(L, 2);
	string ori_path = lua_tostring(L, 3);
	string output_path = lua_tostring(L, 4);
	
	lua_pushstring(L, "gi");
	

	return 1;
}


static luaL_reg liblabel[] = {
	{ "l_label" , l_label },
	{ NULL, NULL }
};


extern "C" int luaopen_liblabel(lua_State * L){
	luaL_register(L, "liblabel", liblabel); // lua 5.1
	return 1;
}
