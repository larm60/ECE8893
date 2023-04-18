#include <iostream>
#include <fstream>
#include <cmath>

//#include "conv.h"  //uncomment this when compiling on vivado

using namespace std;

float conv_layer_input_feature_map1[1][1][187];
float conv_layer_weights1[32][1][5];
float conv_layer_bias1[32];
float conv_layer_input_feature_map2[1][32][187];
float conv_layer_weights2[32][32][5];
float conv_layer_bias2[32];
float conv_layer_input_feature_map3[1][32][187];
float conv_layer_weights3[32][32][5];
float conv_layer_bias3[32];

void read_bin_files()
{
    // Input Feature Map
    ifstream ifs_conv_input("bin/conv1.bin", ios::in | ios::binary);
    ifs_conv_input.read((char*)(**conv_layer_input_feature_map1), 1*1*187*sizeof(float)); //adjust dimensions 3x736x1280
    ifs_conv_input.close();

    // Typecast to fixed-point 
    // for(int c = 0; c < 3; c++)
    //     for(int i = 0; i < 736; i++)
    //         for(int j = 0; j < 1280; j++)
    //             fixp_conv_layer_input_feature_map[c][i][j] = (fm_t) conv_layer_input_feature_map[c][i][j];    
    
    // Weights
    ifstream ifs_conv_weights("bin/conv1_weights.bin", ios::in | ios::binary);
    ifs_conv_weights.read((char*)(**conv_layer_weights1), 32*1*5*sizeof(float));
    ifs_conv_weights.close();
    
    // Typecast to fixed-point 
    // for(int f = 0; f < 64; f++)
    //     for(int c = 0; c < 3; c++)
    //         for(int m = 0; m < 7; m++)
    //             for(int n =0; n < 7; n++)
    //                 fixp_conv_layer_weights[f][c][m][n] = (wt_t) conv_layer_weights[f][c][m][n];
    
    // Bias
    ifstream ifs_conv_bias("bin/conv1_bias.bin", ios::in | ios::binary);
    ifs_conv_bias.read((char*)(conv_layer_bias1), 32*sizeof(float));
    ifs_conv_bias.close();
    
    // // Typecast to fixed-point 
    // for(int f = 0; f < 64; f++)
    //     fixp_conv_layer_bias[f] = (wt_t) conv_layer_bias[f];

    ifstream ifs_conv_input2("bin/conv2.bin", ios::in | ios::binary);
    ifs_conv_input2.read((char*)(**conv_layer_input_feature_map2), 1*32*187*sizeof(float)); //adjust dimensions 3x736x1280
    ifs_conv_input2.close();

    ifstream ifs_conv_weights2("bin/conv2_weights.bin", ios::in | ios::binary);
    ifs_conv_weights2.read((char*)(**conv_layer_weights2), 32*32*5*sizeof(float));
    ifs_conv_weights2.close();

    ifstream ifs_conv_bias2("bin/conv2_bias.bin", ios::in | ios::binary);
    ifs_conv_bias2.read((char*)(conv_layer_bias2), 32*sizeof(float));
    ifs_conv_bias2.close();

    ifstream ifs_conv_input3("bin/conv3.bin", ios::in | ios::binary);
    ifs_conv_input2.read((char*)(**conv_layer_input_feature_map3), 1*32*187*sizeof(float)); //adjust dimensions 3x736x1280
    ifs_conv_input2.close();

    ifstream ifs_conv_weights3("bin/conv3_weights.bin", ios::in | ios::binary);
    ifs_conv_weights2.read((char*)(**conv_layer_weights3), 32*32*5*sizeof(float));
    ifs_conv_weights2.close();

    ifstream ifs_conv_bias3("bin/conv3_bias.bin", ios::in | ios::binary);
    ifs_conv_bias2.read((char*)(conv_layer_bias3), 32*sizeof(float));
    ifs_conv_bias2.close();

}

int main(){
    read_bin_files();
    // for(int i = 0; i < 187; i++){
    //     cout << conv_layer_input_feature_map[0][0][i] << " ";
    // }

    for(int i = 0; i < 32; i++){
        for(int j = 0; j < 5; j++) {
            cout << conv_layer_weights1[i][0][j] << " ";
        }
    }

    // for(int i = 0; i < 32; i++){
    //     cout << conv_layer_bias[i] << " ";
    // }
}