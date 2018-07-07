#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>

#include <vector>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define DEBUG


void label(string path, string type) {
	// load
	Mat im = imread(path, CV_LOAD_IMAGE_GRAYSCALE);

	// preparation
	blur(im, im, Size(25, 25));
	threshold(im, im, 0, 255, CV_THRESH_OTSU);

	// find contours
	vector< vector< Point > > contours;
	vector< Vec4i > hierarchy;
	findContours(im, contours, hierarchy, RETR_EXTERNAL,CHAIN_APPROX_NONE, Point());

#ifdef DEBUG
	// load original im
	string ori_path = "/data8T/aucid/guideDogBackend/input_image/image/img.jpg";
	Mat ori_im = imread(ori_path);
#endif

	// save rect points
	Point2f P[4];
	int n = contours.size();
	for (int i = 0; i < n; i ++) {
		RotatedRect rect = minAreaRect(contours[i]);
		rect.points(P);

#ifdef DEBUG
		drawContours(ori_im, contours, i, Scalar(255), 1, 8, hierarchy);
		for (int j = 0; j < 4; j ++)
			line(ori_im,P[j], P[(j + 1) % 4], Scalar(255), 2);
#endif
	}

#ifdef DEBUG
	putText(ori_im, type, P[1], FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 3, 3);
	imwrite("labeled.png", ori_im);
#endif
}


int l_label(lua_State * L) {
	string path = luaL_checkstring(L, 1);
	//string type = luaL_checkstring(L, 2);

	return 1;
}


static const struct luaL_reg liblabel[] = {
	{ "l_label" , l_label },
	{ NULL, NULL }
};


int luaopen_liblabel(lua_State * L){
#if LUA_VERSION_NUM >= 502
	lua_newtable(L);
	//luaL_newlib(L, liblabel); // 5.2
	luaL_setfuncs (L, liblabel, 0);
#else
	luaL_register(L, "liblabel", liblabel); // lua 5.1
#endif
	return 1;
}


/*
int main() {
	//label("../output2/output-Road.png", "road");
}
*/
