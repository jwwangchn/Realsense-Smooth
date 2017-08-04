#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>


// 滤波窗口大小 WINDOWS_SIZE - 1 为 4 的倍数
#define WINDOWS_SIZE 13


// 内层数量 = (WINDOWS_SIZE / 2 + 1) * (WINDOWS_SIZE / 2 + 1) - 1
// 外层数量 = WINDOWS_SIZE * WINDOWS_SIZE - ((WINDOWS_SIZE / 2 + 1) * (WINDOWS_SIZE / 2 + 1) - 1)

#define INNER_NUMBER (WINDOWS_SIZE / 2 + 1) * (WINDOWS_SIZE / 2 + 1) - 1
#define OUTER_NUMBER WINDOWS_SIZE * WINDOWS_SIZE - ((WINDOWS_SIZE / 2 + 1) * (WINDOWS_SIZE / 2 + 1))

// 内外层阈值
#define INNER_BAND_THRESHOLD INNER_NUMBER / 2 - 1
#define OUTER_BAND_THRESHOLD OUTER_NUMBER / 2 - 1

using namespace cv;
using namespace std;

int IMAGE_HEIGHT;
int IMAGE_WIDTH;

Mat realSenseSmooth(Mat i_depth)
{
	cout << "INNER_NUMBER: " << INNER_NUMBER << " OUTER_NUMBER: " << OUTER_NUMBER << endl;
	double minv = 0.0, maxv = 0.0;
    double* minp = &minv;
    double* maxp = &maxv;

    minMaxIdx(i_depth,minp,maxp);

    cout << "Mat minv = " << minv << endl;
    cout << "Mat maxv = " << maxv << endl;
//    Size size(512,424);
//    resize(i_depth,i_depth,size);

	IMAGE_HEIGHT = i_depth.rows;
	IMAGE_WIDTH = i_depth.cols;

    cout << IMAGE_WIDTH << " " << IMAGE_HEIGHT<<endl;

	Mat i_before(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC4);  // 为了显示方便
	Mat i_after(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC4);   // 为了显示方便
	Mat i_result(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1); // 滤波结果

    unsigned int maxDepth = 0;
	//	unsigned short* depthArray = (unsigned short*)i_depth.data;
	unsigned char depthArray[IMAGE_HEIGHT * IMAGE_WIDTH];


	unsigned int iZeroCountBefore = 0;
	unsigned int iZeroCountAfter = 0;
	for (int i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT; i++)
	{
		int row = i / IMAGE_WIDTH;
		int col = i % IMAGE_WIDTH;
		depthArray[i] = i_depth.at<uint8_t>(row, col);
		unsigned char depthValue = depthArray[i];
		//        unsigned short depthValue = depthArray[row * IMAGE_WIDTH + col];
		// if (depthValue == 0)
		// {
		// 	i_before.data[i * 4] = 255;
		// 	i_before.data[i * 4 + 1] = 0;
		// 	i_before.data[i * 4 + 2] = 0;
		// 	i_before.data[i * 4 + 3] = depthValue / 256;
		// 	iZeroCountBefore++;
		// }
		// else
		// {
		// 	i_before.data[i * 4] = depthValue / 8000.0f * 256;
		// 	i_before.data[i * 4 + 1] = depthValue / 8000.0f * 256;
		// 	i_before.data[i * 4 + 2] = depthValue / 8000.0f * 256;
		// 	i_before.data[i * 4 + 3] = depthValue / 8000.0f * 256;
		// }
		maxDepth = depthValue > maxDepth ? depthValue : maxDepth;
	}
	cout << "max depth value: " << maxDepth << endl;

	// 2. 像素滤波

	// 滤波后深度图的缓存
	unsigned char *smoothDepthArray = (unsigned char *)i_result.data;
	// 我们用这两个值来确定索引在正确的范围内
	int widthBound = IMAGE_WIDTH - 1;
	int heightBound = IMAGE_HEIGHT - 1;

	// 内（8个像素）外（16个像素）层阈值
	int innerBandThreshold = INNER_BAND_THRESHOLD;
	int outerBandThreshold = OUTER_BAND_THRESHOLD;

	// 处理每行像素
	for (int depthArrayRowIndex = 0; depthArrayRowIndex < IMAGE_HEIGHT; depthArrayRowIndex++)
	{
		// 处理一行像素中的每个像素
		for (int depthArrayColumnIndex = 0; depthArrayColumnIndex < IMAGE_WIDTH; depthArrayColumnIndex++)
		{
			int depthIndex = depthArrayColumnIndex + (depthArrayRowIndex * IMAGE_WIDTH);

			// 我们认为深度值为0的像素即为候选像素
			if (depthArray[depthIndex] == 0)
			{
				// 通过像素索引，我们可以计算得到像素的横纵坐标
				int x = depthIndex % IMAGE_WIDTH;
				int y = (depthIndex - x) / IMAGE_WIDTH;

				// filter collection 用来计算滤波器内每个深度值对应的频度，在后面
				// 我们将通过这个数值来确定给候选像素一个什么深度值。
				unsigned char filterCollection[WINDOWS_SIZE*WINDOWS_SIZE - 1][2] = {0};

				// 内外层框内非零像素数量计数器，在后面用来确定候选像素是否滤波
				int innerBandCount = 0;
				int outerBandCount = 0;

				// 下面的循环将会对以候选像素为中心的5 X 5的像素阵列进行遍历。这里定义了两个边界。如果在
				// 这个阵列内的像素为非零，那么我们将记录这个深度值，并将其所在边界的计数器加一，如果计数器
				// 高过设定的阈值，那么我们将取滤波器内统计的深度值的众数（频度最高的那个深度值）应用于候选
				// 像素上
				for (int yi = -(WINDOWS_SIZE / 2); yi < WINDOWS_SIZE / 2 + 1; yi++)
				{
					for (int xi = -(WINDOWS_SIZE / 2); xi < WINDOWS_SIZE / 2 + 1; xi++)
					{
						// yi和xi为操作像素相对于候选像素的平移量

						// 我们不要xi = 0&&yi = 0的情况，因为此时操作的就是候选像素
						if (xi != 0 || yi != 0)
						{
							// 确定操作像素在深度图中的位置
							int xSearch = x + xi;
							int ySearch = y + yi;

							// 检查操作像素的位置是否超过了图像的边界（候选像素在图像的边缘）
							if (xSearch >= 0 && xSearch <= widthBound &&
								ySearch >= 0 && ySearch <= heightBound)
							{
								int index = xSearch + (ySearch * IMAGE_WIDTH);
								// 我们只要非零量
								if (depthArray[index] != 0)
								{
									// 计算每个深度值的频度
									for (int i = 0; i < WINDOWS_SIZE; i++)
									{
										if (filterCollection[i][0] == depthArray[index])
										{
											// 如果在 filter collection中已经记录过了这个深度
											// 将这个深度对应的频度加一
											filterCollection[i][1]++;
											break;
										}
										else if (filterCollection[i][0] == 0)
										{
											// 如果filter collection中没有记录这个深度
											// 那么记录
											filterCollection[i][0] = depthArray[index];
											filterCollection[i][1]++;
											break;
										}
									}

									// 确定是内外哪个边界内的像素不为零，对相应计数器加一
									if (yi <= WINDOWS_SIZE / 2 && yi != -(WINDOWS_SIZE / 2) && xi != WINDOWS_SIZE / 2 && xi != -(WINDOWS_SIZE / 2))
										innerBandCount++;
									else
										outerBandCount++;
								}
							}
						}
					}
				}

				// 判断计数器是否超过阈值，如果任意层内非零像素的数目超过了阈值，
				// 就要将所有非零像素深度值对应的统计众数
				if (innerBandCount >= innerBandThreshold || outerBandCount >= outerBandThreshold)
				{
					char frequency = 0;
					char depth = 0;
					// 这个循环将统计所有非零像素深度值对应的众数
					for (int i = 0; i < WINDOWS_SIZE * WINDOWS_SIZE - 1; i++)
					{
						// 当没有记录深度值时（无非零深度值的像素）
						if (filterCollection[i][0] == 0)
							break;
						if (filterCollection[i][1] > frequency)
						{
							depth = filterCollection[i][0];
							frequency = filterCollection[i][1];
						}
					}

					smoothDepthArray[depthIndex] = depth;
				}
				else
				{
					smoothDepthArray[depthIndex] = 0;
				}
			}
			else
			{
				// 如果像素的深度值不为零，保持原深度值
				smoothDepthArray[depthIndex] = depthArray[depthIndex];
			}
		}
	}

	for (int i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT; i++)
	{
		int row = i / IMAGE_WIDTH;
		int col = i % IMAGE_WIDTH;

		unsigned char depthValue = smoothDepthArray[row * IMAGE_WIDTH + col];
		// if (depthValue == 0)
		// {
		// 	i_after.data[i * 4] = 255;
		// 	i_after.data[i * 4 + 1] = 0;
		// 	i_after.data[i * 4 + 2] = 0;
		// 	i_after.data[i * 4 + 3] = depthValue / 256;
		// 	iZeroCountAfter++;
		// }
		// else
		// {
		// 	i_after.data[i * 4] = depthValue / 8000.0f * 256;
		// 	i_after.data[i * 4 + 1] = depthValue / 8000.0f * 256;
		// 	i_after.data[i * 4 + 2] = depthValue / 8000.0f * 256;
		// 	i_after.data[i * 4 + 3] = depthValue / 8000.0f * 256;
		// }
	}
	cout << "iZeroCountBefore:    " << iZeroCountBefore << "  depthArray[0]:  " << depthArray[0] << endl;
	cout << "iZeroCountAfter:    " << iZeroCountAfter << "  smoothDepthArray[0]:  " << smoothDepthArray[0] << endl;
	return i_result;
}



int main()
{
	// 1. 深度图并显示
	Mat i_depth(IMAGE_HEIGHT, IMAGE_WIDTH, CV_16UC1);
	i_depth = imread("../depth0.png", IMREAD_ANYDEPTH);
	cout << sizeof(i_depth) << endl;

    Mat i_result = realSenseSmooth(i_depth);

	// 3. 显示
	//	thread th = std::thread([&]{
	//		while (true)
	//		{
	imshow("原始深度图", i_depth);
	waitKey(1);
	// imshow("便于观察的原始深度图", i_before);
	// waitKey(1);
	//		}
	//	});

	//	thread th2 = std::thread([&]{
	//		while (true)
	{
		imshow("结果图", i_result);
		waitKey(1);
		// imshow("便于观察的滤波深度图", i_after);
		// waitKey(1);
	}
	//	});
	
	//	th.join();
	//	th2.join();
	waitKey(0);
	return 0;
}