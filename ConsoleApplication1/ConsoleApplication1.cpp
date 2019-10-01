#if 1
#include <opencv2/opencv.hpp>
#include <iostream>
#include "math.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	Mat input_image;
	input_image = imread("E:\\2.jpg");
	if (input_image.data == NULL) {
		return -1; cout << "can't open image.../";
	}
	cout << "input_image rows is " << input_image.rows << endl;
	cout << "input_image cols is " << input_image.cols << endl;
	//-----------------------------------------------------------------------------------//
	//------------------------------梯度检测图像--------------------------------------//
	//-----------------------------------------------------------------------------------//
	Mat input_image1;
	input_image.copyTo(input_image1);
	cvtColor(input_image1, input_image1, CV_BGR2GRAY);
	input_image1.convertTo(input_image1, CV_32FC1);
	Mat sobelx = (Mat_<float>(3, 3) << -0.125, 0, 0.125, -0.25, 0, 0.25, -0.125, 0, 0.125);
	filter2D(input_image1, input_image1, input_image1.type(), sobelx);	//对图像进行卷积操作
	imshow("filter2D", input_image1);
	Mat mul_image;
	multiply(input_image1, input_image1, mul_image);	//矩阵相乘 -> mul_image.
	//imshow("input_image1xinput_image1", input_image1);
	//imshow("mul_image", mul_image);
	const int scaleValue = 4;
	cout << "mean(mul_image) is " << mean(mul_image) << endl;
	cout << "mean(mul_image).val[0] is " << mean(mul_image).val[0] << endl;
	double threshold = scaleValue * mean(mul_image).val[0];//4 * img_s的平均值
	Mat resultImage = Mat::zeros(mul_image.size(), mul_image.type());
	float* pDataimg_s = (float*)(mul_image.data);
	float* pDataresult = (float*)(resultImage.data);
	const int height = input_image1.rows;
	const int width = input_image1.cols;
	//--- 非极大值抑制 + 阈值分割
	cout << "height is " << height << endl;	//636
	cout << "width is " << width << endl;	//875
	for (size_t i = 1; i < height - 1; i++)
	{
		for (size_t j = 1; j < width - 1; j++)
		{
			/*
				i = 1;i*height = 636,列极大值
				假设
				a1 b1 c1 d1 e1
				a2 b2 c2 d2 e2
				a3 b3 c3 d3 e3
				a4 b4 c4 d4 e4
				a5 b5 c5 d5 e5
				pDataimg_s[i*height+j] = b2.
				pDataimg_s[i*height+j-1] = b1
				pDataimg_s[i*height+j+1] = b3
				pDataimg_s[(i-1)*height+j] = a2
				pDataimg_s[(i+1)*height+j] = c2
				(b1&&b2),true,也就是b2>b1且b2>b3
				(b3&&b4),true,也就是b2>a2且b2>c2
				也就是说明当前这点，至少是相邻行的最大值或者相邻列的最大值
				还有要大于阈值.
			*/
			bool b1 = (pDataimg_s[i * height + j] > pDataimg_s[i * height + j - 1]);
			bool b2 = (pDataimg_s[i * height + j] > pDataimg_s[i * height + j + 1]);
			bool b3 = (pDataimg_s[i * height + j] > pDataimg_s[(i - 1) * height + j]);
			bool b4 = (pDataimg_s[i * height + j] > pDataimg_s[(i + 1) * height + j]);
			
			pDataresult[i * height + j] = 255 * ((pDataimg_s[i * height + j] > threshold) && ((b1 && b2) || (b3 && b4)));
		}
	}
	resultImage.convertTo(resultImage, CV_8UC1);
	imshow("resultImage", resultImage);
	//-----------------------------------------------------------------------------------//
	//---------------------------------HSV通道提取---------------------------------------//
	//-----------------------------------------------------------------------------------//
	Mat input_image2;
	input_image.copyTo(input_image2);
	Mat img_h, img_s, img_v, img_hsv;
	cvtColor(input_image2, img_hsv, CV_BGR2HSV);
	//imshow("img_hsv", img_hsv);
	vector<Mat> hsv_vec;
	split(img_hsv, hsv_vec);
	img_h = hsv_vec[0];
	img_s = hsv_vec[1];
	img_v = hsv_vec[2];
	imshow("img_h0", img_h);
	cout << "img_h0 " << img_h.type() << endl;
	img_h.convertTo(img_h, CV_32F);
	img_s.convertTo(img_s, CV_32F);
	img_v.convertTo(img_v, CV_32F);
	cout << "img_hh " << img_h.type() << endl;
	imshow("img_h", img_h);
	normalize(img_h, img_h, 0, 1, NORM_MINMAX);
	normalize(img_s, img_s, 0, 1, NORM_MINMAX);
	normalize(img_v, img_v, 0, 1, NORM_MINMAX);
	imshow("img_h1", img_h);
	imshow("img_s1", img_s);
	imshow("img_v1", img_v);
	Mat test = img_v > 0.5;
	imshow("test", test);
	/*
		h 0-180	色调
		s 0-255 色饱和度
		v 0-255	饱和度
		blue通道.
		h:100-124,s:43-,v:46.

	*/
	//Mat img_vblue = ((img_h > 0.45) & (img_h < 0.75) & (img_s > 0.15) & (img_v > 0.25));//蓝色通道提取
	Mat img_vblue = ((img_h > 0.55) & (img_h < 0.69) & (img_s > 0.17) & (img_v > 0.18));//蓝色通道提取
	imshow("img_vblue", img_vblue);	//img_vblue蓝色为255部分。其余为0.

	Mat vbule_gradient = Mat::zeros(input_image2.size(), CV_8UC1);
	for (size_t i = 1; i < height - 1; i++)
	{
		for (size_t j = 1; j < width - 1; j++)
		{
			/*
				Rect rec;
				rec.x = j - 1;
				rec.y = i - 1;
				rec.width  = 3;
				rec.height = 3;
				resultImage是非极大值和阈值的结果.
				img_vblue 是选择的蓝色区域.
				如果都是255(假设)，那么设置为1。
				应该类似，边缘+蓝色区域 
			*/
			//----梯度和蓝色区域重合的部分，也可以用上面的矩形3X3的判断
			vbule_gradient.at<uchar>(i, j) = (resultImage.at<uchar>(i, j) == 255 && img_vblue.at<uchar>(i, j) != 0) ? 255 : 0;
			//vbule_gradient.at<uchar>(i, j) = (resultImage.at<uchar>(i, j) == 255 && countNonZero(img_vblue(rec)) >= 1) ? 255 : 0;
		}
	}
	imshow("vbule_gradient", vbule_gradient);
	//-----------------------------------------------------------------------------------//
	//-----------------------------形态学+轮廓提取车牌-----------------------------------//
	//-----------------------------------------------------------------------------------//
	Mat morph;
	morphologyEx(vbule_gradient, morph, MORPH_CLOSE, Mat::ones(2, 25, CV_8UC1));
	//imshow("morph", morph);	
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(morph, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(-1, -1));
	Rect rec_adapt;
	for (size_t i = 0; i < contours.size(); i++)
	{
		//----矩形区域非零像素占总的比例，防止有较大的空白区域干扰检测结果
		//----矩形的长宽限制，也可以再增加额外条件：长宽比例等
		//countNonZero 可以得到非零像素点的个数。
		//boundingRect 计算轮廓的最小外接矩形
		
		int true_pix_count = countNonZero(morph(boundingRect(contours[i])));
		double true_pix_rate = static_cast<double>(true_pix_count) / static_cast<double>(boundingRect(contours[i]).area());
		if (boundingRect(contours[i]).height > 10 && boundingRect(contours[i]).width > 80 && true_pix_rate > 0.7)
		{
			rec_adapt = boundingRect(contours[i]);
			drawContours(morph, contours, static_cast<int>(i), Scalar(200, 200, 0), 2);
		}
	}
	imshow("morph1", morph);
	imshow("Area Brand", input_image(rec_adapt));
	Mat output_image;
	resize(input_image(rec_adapt), output_image, Size(136, 36));
	imshow("resize", output_image);
	waitKey(0);
	return 0;
}
#endif