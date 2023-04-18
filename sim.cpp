///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////

// KENTA'S CODE:

#include <iostream>
#include <fstream>
#include <cmath>

#include "conv.h"  //uncomment this when compiling on vivado

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


//Extracode
float conv1_real_output[1][32][187];
float conv1_output[1][32][187];
float max_pool1[1][32][92];
float real_max_pool1[1][32][92];
float input[1][1][187];

wt_t fixed_conv_layer_weights1[32][1][5];
wt_t fixed_conv_layer_bias1[32];

void read_bin_files()
{
    // Input Feature Map  ***UNCOMMENT
    // ifstream ifs_conv_input("../binProject/conv1.bin", ios::in | ios::binary);
    // ifs_conv_input.read((char*)(**conv_layer_input_feature_map1), 1*1*187*sizeof(float)); //adjust dimensions 3x736x1280
    // ifs_conv_input.close();

    // Typecast to fixed-point 
    // for(int c = 0; c < 3; c++)
    //     for(int i = 0; i < 736; i++)
    //         for(int j = 0; j < 1280; j++)
    //             fixp_conv_layer_input_feature_map[c][i][j] = (fm_t) conv_layer_input_feature_map[c][i][j];    
    
    // Weights
    ifstream ifs_conv_weights1("../binProject/conv1_weights.bin", ios::in | ios::binary);
    ifs_conv_weights1.read((char*)(**conv_layer_weights1), 32*1*5*sizeof(float));
    ifs_conv_weights1.close();
    
    // Typecast to fixed-point 
    for(int f = 0; f < 32; f++)
        for(int c = 0; c < 1; c++)
            for(int m = 0; m < 5; m++)
                    fixed_conv_layer_weights1[f][c][m] = (wt_t) conv_layer_weights1[f][c][m];
    
    // Bias ***UNCOMMENT
    ifstream ifs_conv_bias1("../binProject/conv1_bias.bin", ios::in | ios::binary);
    ifs_conv_bias1.read((char*)(conv_layer_bias1), 32*sizeof(float));
    ifs_conv_bias1.close();
    
    // // Typecast to fixed-point 
    // for(int f = 0; f < 32; f++)
    //     fixed_conv_layer_bias1[f] = (wt_t) conv_layer_bias1[f];

    // ***UNCOMMENT
    // ifstream ifs_conv_input2("../binProject/conv2.bin", ios::in | ios::binary);
    // ifs_conv_input2.read((char*)(**conv_layer_input_feature_map2), 1*32*187*sizeof(float)); //adjust dimensions 3x736x1280
    // ifs_conv_input2.close();

    // ifstream ifs_conv_weights2("../binProject/conv2_weights.bin", ios::in | ios::binary);
    // ifs_conv_weights2.read((char*)(**conv_layer_weights2), 32*32*5*sizeof(float));
    // ifs_conv_weights2.close();

    // ifstream ifs_conv_bias2("../binProject/conv2_bias.bin", ios::in | ios::binary);
    // ifs_conv_bias2.read((char*)(conv_layer_bias2), 32*sizeof(float));
    // ifs_conv_bias2.close();

    // ifstream ifs_conv_input3("../binProject/conv3.bin", ios::in | ios::binary);
    // ifs_conv_input2.read((char*)(**conv_layer_input_feature_map3), 1*32*187*sizeof(float)); //adjust dimensions 3x736x1280
    // ifs_conv_input2.close();

    // ifstream ifs_conv_weights3("../binProject/conv3_weights.bin", ios::in | ios::binary);
    // ifs_conv_weights2.read((char*)(**conv_layer_weights3), 32*32*5*sizeof(float));
    // ifs_conv_weights2.close();

    // ifstream ifs_conv_bias3("../binProject/conv3_bias.bin", ios::in | ios::binary);
    // ifs_conv_bias2.read((char*)(conv_layer_bias3), 32*sizeof(float));
    // ifs_conv_bias2.close();

    // Golden Output
    ifstream ifs_golden_output("../binProject/conv1.bin", ios::in | ios::binary);
    ifs_golden_output.read((char*)(**conv1_real_output), 1*32*187*sizeof(float));    
    ifs_golden_output.close();

    // Golden Output
    ifstream ifs_golden_output_max1("../binProject/max1.bin", ios::in | ios::binary);
    ifs_golden_output_max1.read((char*)(**real_max_pool1), 1*32*92*sizeof(float));    
    ifs_golden_output_max1.close();

}

int main(){
    
    // Initialize input with ones
    for (int i = 0; i < 1; i++)
    {
        for (int j = 0; j < 1; j++)
        {
            for (int k = 0; k < 187; k++)
            {
                input[i][j][k] = 1.0f;
            }
        }
    }

    read_bin_files();
    std::cout << "Bin files read!\n" << std::endl;

    //*** PRINT Weights and bias for first layer
    for (int i = 0; i < 32; i++) // Check first channel output
    {
        std::cout << "Bias channel:  " << conv_layer_bias1[i] << std::endl;
        std::cout << "Weights channel:  " << i << std::endl;
        for (int j = 0; j < 5; j++)
        {
            std::cout << conv_layer_weights1[i][0][j] << std::endl;
        }
    }

    conv_1d_start( //conv1 code
    conv1_output, //ignore first dimension, 32 output channels, 187 signal length
    input, //ignore first dimension, 1 input channel, signal length
    conv_layer_weights1, //32 out channels, kernel size:5
    conv_layer_bias1 //32 outchannels
    );

    std::cout << "First conv!\n" << std::endl;

    max_pooling1(
    conv1_output,
    max_pool1
    );

    std::cout << "First Max pool!\n" << std::endl;

    // for(int i = 0; i < 187; i++){
    //     std::cout << input[0][0][i] << "\n" << std::endl;
    // }


    // for(int i = 0; i < 32; i++){
    //     for(int j = 0; j < 5; j++) {
    //         cout << conv_layer_weights[i][0][j] << " ";
    //     }
    // }

    // for(int i = 0; i < 32; i++){
    //     cout << conv_layer_bias[i] << " ";
    // }


        //*** CODE TO PRINT Weights and bias for 1 layer
        // for(int i = 0; i < 32; i++) //Check first channel output
        // {
        //     std::cout << "Bias channel:  " << conv_layer_bias1[i] << std::endl;
        //     std::cout << "Weights channel:  " << i << std::endl;
        //     for(int j = 0; j < 5; j++)
        //     {
        //         std::cout << conv_layer_weights1[i][0][j] << std::endl;
        //     }
        // }



        // *** CODE TO PRINT MSE of first layer and expected/actual outputs
        // long double mse = 0.0;
        // for(int i = 0; i < 32; i++) //Check first channel output
        // {
        //     for(int j = 0; j < 187; j++)
        //     {
        //         mse += std::pow((conv1_real_output[0][i][j] 
        //                               - conv1_output[0][i][j]), 2);
            
        //         std::cout << "Expected output:  " << conv1_real_output[0][i][j] << std::endl;
        //         std::cout << "Real output:  " << conv1_output[0][i][j] << std::endl;
        //     }
        // }
        // mse = mse / (32*187);
        // std::cout << "Output MSE:  " << mse << std::endl;

        //*** CODE TO PRINT Max pooling outputs and MSE:
        long double mse = 0.0;
        for(int i = 0; i < 32; i++) //32: But first I'm checking the first channel outputs
        {
            for(int j = 0; j < 92; j++)
            {
                mse += std::pow((max_pool1[0][i][j] 
                                      - real_max_pool1[0][i][j]), 2);
            
                std::cout << "Expected output:  " << real_max_pool1[0][i][j] << std::endl;
                std::cout << "Real output:  " << max_pool1[0][i][j] << std::endl;
            }
        }
        mse = mse / (32*92);
        std::cout << "Output MSE:  " << mse << std::endl;
        //End *** Max pooling outputs and MSE
        
}
