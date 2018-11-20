#include "pch.h"
#include <iostream>
// 1.1 RGB转换为YUV
void RGB2YUV(double R, double G, double B, double &Y, double &U, double &V)
{
	Y = static_cast<double>(0.257*R + 0.504*G + 0.098*B + 16); //计算Y
	U = static_cast<double>(-0.148*R - 0.291*G + 0.439*B + 128); //计算U
	V = static_cast<double>(0.439*R - 0.368*G - 0.071*B + 128); //计算V
}

// 1.2 YUV转换为RGB 
void YUV2RGB(double Y, double U, double V, double &R, double &G, double &B)
{
	B = static_cast<double>(1.164*(Y - 16) + 2.018*(U - 128)); //计算B
	G = static_cast<double>(1.164*(Y - 16) - 0.391*(U - 128) - 0.813*(V - 128)); //计算G
	R = static_cast<double>(1.164*(Y - 16) + 1.596*(V - 128)); //计算 R
}



// 2.1 RGB转换为HSV

void RGB2HSV(float R, float G, float B, float &H, float &S, float V)
{
	float R_1, G_1, B_1;
	R_1 = (R / 255);                    //将(R,G,B)分别转换成0到1之间的实数
	G_1 = (G / 255);
	B_1 = (B / 255);

	// H取值范围为 [0,360], S取值范围为 [0,1], V取值范围为 [0,1]

	float min_value, max_value, delta;
	min_value = min(min(R_1, G_1), B_1);  //求R_1, G_1，B_1最小值
	max_value = max(max(R_1, G_1), B_1);  //求R_1, G_1，B_1最大值
	delta = max_value - min_value;   //求最大值与最小值差值

	float H, S, V;
	V = max_value;  //计算V值
	if (max_value != 0)
		S = delta / max_value; // 计算S值
	else
	{
		// s = 0, v is undefined
		S = 0;
		H = UNDEFINEDCOLOR;
		return;
	}
	if (delta == 0)
		H = 0;
	else if (R == max_value)
		H = (G - B) / delta; // (G-B)/(max-min)
	else if (G == max_value)
		H = 2 + (B - R) / delta; // 120+(B-R)/(max-min)*60,此处没有乘60
	else
		H = 4 + (R - G) / delta; // 240+(R-G)/(max-min)*60,此处没有乘60
	H *= 60; //统一乘上60
	if (H < 0)
		H += 360;
}


// 2.2 HSV转换为RGB
void HSV2RGB(double H, double S, double V, double &R, double &G, double &B)
{
	double R, G, B;
	if (S == 0)
	{
		R = V * 255.0f;               //将RGB值转换成0到255之间
		G = V * 255.0f;
		B = V * 255.0f;
	}
	else
	{
		double h, h_i, p, q, t;
		h = H / 60.0;

		h_i = floor(h);
		p = V * (1 - S);
		q = V * (1 - S * (h - h_i));
		t = V * (1 - S * (1 - (h - h_i)));

		double r, g, b;
		if (h_i == 0)
		{
			r = V;                     //h_i=0,(r,g,b)=(v,t,p)
			g = t;
			b = p;
		}
		else if (h_i == 1)
		{
			r = q;                     //h_i=1,(r,g,b)=(q,v,p)
			g = V;
			b = p;
		}
		else if (h_i == 2)
		{
			r = p;                     //h_i=2,(r,g,b)=(p,v,t)
			g = V;
			b = t;
		}
		else if (h_i == 3)
		{
			r = p;                     //h_i=3,(r,g,b)=(p,q,v)
			g = q;
			b = V;
		}
		else if (h_i == 4)
		{
			r = t;                     //h_i=4,(r,g,b)=(t,p,v)
			g = p;
			b = V;
		}
		else
		{
			r = V;                     //h_i=5,(r,g,b)=(v,p,q)
			g = p;
			b = q;
		}

		R = r * 255.0f;                 //将RGB值转换成0到255之间
		G = g * 255.0f;
		B = b * 255.0f;
	}
}