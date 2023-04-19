#include <iostream>
#include <fstream>
#include <cmath>
#include "conv.h"

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

float conv_layer_output_feature_map[1][32][187];

void read_bin_files()
{
    // Input Feature Map
    ifstream ifs_conv_input("conv1.bin", ios::in | ios::binary);
    ifs_conv_input.read((char*)(**conv_layer_input_feature_map1), 1*32*187*sizeof(float)); //adjust dimensions 3x736x1280
    ifs_conv_input.close();

    // Typecast to fixed-point 
    // for(int c = 0; c < 3; c++)
    //     for(int i = 0; i < 736; i++)
    //         for(int j = 0; j < 1280; j++)
    //             fixp_conv_layer_input_feature_map[c][i][j] = (fm_t) conv_layer_input_feature_map[c][i][j];    
    
    // Weights
    ifstream ifs_conv_weights("conv1_weights.bin", ios::in | ios::binary);
    ifs_conv_weights.read((char*)(**conv_layer_weights1), 32*1*5*sizeof(float));
    ifs_conv_weights.close();
    
    // Typecast to fixed-point 
    // for(int f = 0; f < 64; f++)
    //     for(int c = 0; c < 3; c++)
    //         for(int m = 0; m < 7; m++)
    //             for(int n =0; n < 7; n++)
    //                 fixp_conv_layer_weights[f][c][m][n] = (wt_t) conv_layer_weights[f][c][m][n];
    
    // Bias
    ifstream ifs_conv_bias("../bin/conv1_bias.bin", ios::in | ios::binary);
    ifs_conv_bias.read((char*)(conv_layer_bias1), 32*sizeof(float));
    ifs_conv_bias.close();
    
    // // Typecast to fixed-point 
    // for(int f = 0; f < 64; f++)
    //     fixp_conv_layer_bias[f] = (wt_t) conv_layer_bias[f];

    ifstream ifs_conv_input2("../bin/conv2.bin", ios::in | ios::binary);
    ifs_conv_input2.read((char*)(**conv_layer_input_feature_map2), 1*32*187*sizeof(float)); //adjust dimensions 3x736x1280
    ifs_conv_input2.close();

    ifstream ifs_conv_weights2("../bin/conv2_weights.bin", ios::in | ios::binary);
    ifs_conv_weights2.read((char*)(**conv_layer_weights2), 32*32*5*sizeof(float));
    ifs_conv_weights2.close();

    ifstream ifs_conv_bias2("../bin/conv2_bias.bin", ios::in | ios::binary);
    ifs_conv_bias2.read((char*)(conv_layer_bias2), 32*sizeof(float));
    ifs_conv_bias2.close();

    ifstream ifs_conv_input3("../bin/conv3.bin", ios::in | ios::binary);
    ifs_conv_input2.read((char*)(**conv_layer_input_feature_map3), 1*32*187*sizeof(float)); //adjust dimensions 3x736x1280
    ifs_conv_input2.close();

    ifstream ifs_conv_weights3("../bin/conv3_weights.bin", ios::in | ios::binary);
    ifs_conv_weights2.read((char*)(**conv_layer_weights3), 32*32*5*sizeof(float));
    ifs_conv_weights2.close();

    ifstream ifs_conv_bias3("../bin/conv3_bias.bin", ios::in | ios::binary);
    ifs_conv_bias2.read((char*)(conv_layer_bias3), 32*sizeof(float));
    ifs_conv_bias2.close();

}

int main(){
    read_bin_files();

    long double mse = 0.0;

    std::cout << "Beginning C model..." << std::endl;

    conv_1d_start ( //conv1 code
        conv_layer_output_feature_map, //ignore first dimension, 32 output channels, 187 signal length
        conv_layer_input_feature_map1, //ignore first dimension, 1 input channel, signal length
        conv_layer_weights1, //32 out channels, kernel size:5
        conv_layer_bias1 //32 outchannels
    );
    
    // conv_1d ( //conv2, conv3 code
    //     conv_layer_output_feature_map, //ignore first dimension, 32 output channels, 187 signal length
    //     conv_layer_input_feature_map2, //ignore first dimension, 1 input channel, signal length
    //     conv_layer_weights2, //32 out channels, kernel size:5
    //     conv_layer_bias2 //32 outchannels
    // );

    std::cout << "C model complete!\n" << std::endl;
    
    // Compute Mean-Squared-Error
    for(int f = 0; f < 1; f++)
    {
        for(int i = 0; i < 32; i++)
        {
            for(int j = 0; j < 187; j++)
            {
                cout << conv_layer_output_feature_map[f][i][j];        
            }
            cout << endl;
        }
        
        #ifdef PRINT_DEBUG
            // Prints sample output values (first feature of each channel) for comparison
            // Modify as required for debugging
            int row = 0;
            int col = 0;
            
            cout << "Output feature[" << f << "][" << row << "][" << col << "]: ";
            cout << "Expected: " << conv_layer_golden_output_feature_map[f][row][col] << "\t"; 
            cout << "Actual: " << conv_layer_output_feature_map[f][row][col];
            cout << endl; 
        #endif
    }
    
    // mse = mse / (1 * 32 * 187);

    // std::cout << "Output MSE:  " << mse << std::endl;
    
    // std::cout << "----------------------------------------" << std::endl;
    // if(mse > 0 && mse < std::exp(-12))
    // {
    //     std::cout << "Simulation SUCCESSFUL!!!" << std::endl;
    // }
    // else
    // {
    //     std::cout << "Simulation FAILED :(" << std::endl;
    // }
    // std::cout << "----------------------------------------" << std::endl;

    return 0;
}