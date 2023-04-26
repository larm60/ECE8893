#include <iostream>
#include <fstream>
#include <cmath>
#include "conv.h" 

using namespace std;

//renamed input_feature_maps to output_feature_maps because the bin is actually the output of the layer
float conv_layer_output_feature_map1_golden[1][32][187];
float conv_layer_weights1[32][1][5];
float conv_layer_bias1[32];

float conv_layer_weights2[32][32][5];
float conv_layer_bias2[32];

float conv_layer_weights3[32][32][5];
float conv_layer_bias3[32];

float conv_layer_weights4[32][32][5];
float conv_layer_bias4[32];

fm_t fixp_conv_layer_output_feature_map1[1][32][187];
fm_t fixp_conv_layer_output_feature_map1_max[1][32][92];
fm_t fixp_conv_layer_output_feature_map2[1][32][92];
fm_t fixp_conv_layer_output_feature_map2_max[1][32][44];
fm_t fixp_conv_layer_output_feature_map3[1][32][44];
fm_t fixp_conv_layer_output_feature_map4[1][32][44];
fm_t fixp_conv_layer_output_feature_map3_max[1][32][20];
fm_t fixp_conv_layer_output_feature_map4_max[1][32][8];
fm_t fixp_conv_layer_output_feature_map5_max[1][32][2];
fm_t fixp_dense1_output[1][32];
fm_t fixp_dense2_output[1][5];

fm_t fixp_conv_layer_output_feature_flat[1][64];

wt_t fixp_conv_layer_weights1[32][1][5];
wt_t fixp_conv_layer_bias1[32];
wt_t fixp_conv_layer_weights2[32][1][5];
wt_t fixp_conv_layer_bias2[32];
wt_t fixp_conv_layer_weights3[32][1][5];
wt_t fixp_conv_layer_bias3[32];
wt_t fixp_conv_layer_weights4[32][1][5];
wt_t fixp_conv_layer_bias4[32];

// declare weights and biases for dense1 and dense2
wt_t fixp_dense1_weights[32][64]; 
wt_t fixp_dense1_bias[32];
wt_t fixp_dense2_weights[5][32];
wt_t fixp_dense2_bias[5];


void read_bin_files()
{    
    // Weights 1
    ifstream ifs_conv1_weights("bin/conv1_weights.bin", ios::in | ios::binary);
    ifs_conv1_weights.read((char*)(**conv_layer_weights1), 32*1*5*sizeof(float));
    ifs_conv1_weights.close();
    
    // Typecast to fixed-point 
    for(int f = 0; f < 32; f++)
        for(int c = 0; c < 1; c++)
            for(int m = 0; m < 5; m++)
                    fixp_conv_layer_weights1[f][c][m] = (wt_t) conv_layer_weights1[f][c][m];
    
    // Bias 1
    ifstream ifs_conv1_bias("bin/conv1_bias.bin", ios::in | ios::binary);
    ifs_conv1_bias.read((char*)(conv_layer_bias1), 32*sizeof(float));
    ifs_conv1_bias.close();
    
    // // Typecast to fixed-point 
    for(int f = 0; f < 32; f++)
        fixp_conv_layer_bias1[f] = (wt_t) conv_layer_bias1[f];

    // Weights 2
    ifstream ifs_conv2_weights("bin/conv2_weights.bin", ios::in | ios::binary);
    ifs_conv2_weights.read((char*)(**conv_layer_weights2), 32*1*5*sizeof(float));
    ifs_conv2_weights.close();
    
    // Typecast to fixed-point 
    for(int f = 0; f < 32; f++)
        for(int c = 0; c < 1; c++)
            for(int m = 0; m < 5; m++)
                    fixp_conv_layer_weights2[f][c][m] = (wt_t) conv_layer_weights2[f][c][m];
    
    // Bias 2
    ifstream ifs_conv2_bias("bin/conv2_bias.bin", ios::in | ios::binary);
    ifs_conv2_bias.read((char*)(conv_layer_bias2), 32*sizeof(float));
    ifs_conv2_bias.close();
    
    // // Typecast to fixed-point 
    for(int f = 0; f < 32; f++)
        fixp_conv_layer_bias2[f] = (wt_t) conv_layer_bias2[f];

    // Weights 3
    ifstream ifs_conv3_weights("bin/conv3_weights.bin", ios::in | ios::binary);
    ifs_conv3_weights.read((char*)(**conv_layer_weights3), 32*1*5*sizeof(float));
    ifs_conv3_weights.close();
    
    // Typecast to fixed-point 
    for(int f = 0; f < 32; f++)
        for(int c = 0; c < 1; c++)
            for(int m = 0; m < 5; m++)
                    fixp_conv_layer_weights3[f][c][m] = (wt_t) conv_layer_weights3[f][c][m];
    
    // Bias 3
    ifstream ifs_conv3_bias("bin/conv3_bias.bin", ios::in | ios::binary);
    ifs_conv3_bias.read((char*)(conv_layer_bias3), 32*sizeof(float));
    ifs_conv3_bias.close();
    
    // // Typecast to fixed-point 
    for(int f = 0; f < 32; f++)
        fixp_conv_layer_bias3[f] = (wt_t) conv_layer_bias3[f];

    // Weights 4
    ifstream ifs_conv4_weights("bin/conv4_weights.bin", ios::in | ios::binary);
    ifs_conv4_weights.read((char*)(**conv_layer_weights4), 32*1*5*sizeof(float));
    ifs_conv4_weights.close();
    
    // Typecast to fixed-point 
    for(int f = 0; f < 32; f++)
        for(int c = 0; c < 1; c++)
            for(int m = 0; m < 5; m++)
                    fixp_conv_layer_weights4[f][c][m] = (wt_t) conv_layer_weights4[f][c][m];
    
    // Bias 4
    ifstream ifs_conv4_bias("bin/conv4_bias.bin", ios::in | ios::binary);
    ifs_conv4_bias.read((char*)(conv_layer_bias4), 32*sizeof(float));
    ifs_conv4_bias.close();
    
    // // Typecast to fixed-point 
    for(int f = 0; f < 32; f++)
        fixp_conv_layer_bias4[f] = (wt_t) conv_layer_bias4[f];


    ifstream ifs_dense1_bias("bin/dense1_bias.bin", ios::in | ios::binary);
    ifs_dense1_bias.read((char*)(fixp_dense1_bias), 32*sizeof(float));
    ifs_dense1_bias.close();

    ifstream ifs_dense2_bias("bin/dense2_bias.bin", ios::in | ios::binary);
    ifs_dense2_bias.read((char*)(fixp_dense2_bias), 32*sizeof(float));
    ifs_dense2_bias.close();


    ifstream ifs_dense1_weights("bin/dense1_weights.bin", ios::in | ios::binary);
    ifs_dense1_weights.read((char*)(fixp_dense1_weights), 32*sizeof(float));
    ifs_dense1_weights.close();


    ifstream ifs_dense2_weights("bin/dense2_weights.bin", ios::in | ios::binary);
    ifs_dense2_weights.read((char*)(fixp_dense2_weights), 32*sizeof(float));
    ifs_dense2_weights.close();

    
}

int main(){
    read_bin_files();

    long double mse = 0.0;

    std::cout << "Beginning C model..." << std::endl;

    fm_t fixp_conv_layer_input_feature_map1[1][1][187] = {1}; //sets input features all to 1

    conv1d_1 ( //conv1 code
        fixp_conv_layer_output_feature_map1, //ignore first dimension, 32 output channels, 187 signal length
        fixp_conv_layer_input_feature_map1, //ignore first dimension, 1 input channel, signal length
        fixp_conv_layer_weights1, //32 out channels, kernel size:5
        fixp_conv_layer_bias1 //32 outchannels
    );

    max_pooling1(fixp_conv_layer_output_feature_map1, fixp_conv_layer_output_feature_map1_max);
    
    conv1d_2 ( //conv2
        fixp_conv_layer_output_feature_map2, 
        fixp_conv_layer_output_feature_map1_max, 
        conv_layer_weights2, //32 out channels, kernel size:5
        conv_layer_bias2 //32 outchannels
    );

    max_pooling2(fixp_conv_layer_output_feature_map2, fixp_conv_layer_output_feature_map2_max);

    conv1d_3_4 ( //conv3
        fixp_conv_layer_output_feature_map3, 
        fixp_conv_layer_output_feature_map2_max, 
        conv_layer_weights2, //32 out channels, kernel size:5
        conv_layer_bias2 //32 outchannels
    );

    conv1d_3_4 ( //conv3
        fixp_conv_layer_output_feature_map4, 
        fixp_conv_layer_output_feature_map3, 
        conv_layer_weights2, //32 out channels, kernel size:5
        conv_layer_bias2 //32 outchannels
    );

    max_pooling3(fixp_conv_layer_output_feature_map4, fixp_conv_layer_output_feature_map3_max);
    max_pooling4(fixp_conv_layer_output_feature_map3_max, fixp_conv_layer_output_feature_map4_max);
    max_pooling5(fixp_conv_layer_output_feature_map4_max, fixp_conv_layer_output_feature_map5_max);

    flatten(fixp_conv_layer_output_feature_map5_max, fixp_conv_layer_output_feature_flat); //makes array into (1,64)

    dense1(fixp_conv_layer_output_feature_flat, fixp_dense1_bias, fixp_dense1_weights,  fixp_dense1_output);
    dense2(fixp_dense1_output, fixp_dense2_bias, fixp_dense2_weights, fixp_dense2_output);

    //need to do linear/dense layers here

    std::cout << "C model complete!\n" << std::endl;

    cout << "fixp_conv_layer_output_feature_flat" << endl;
    for(int f = 0; f < 64; f++){
        cout <<  fixp_conv_layer_output_feature_flat[f] << "   ";
    }
    cout << endl;
    cout << "fixp_dense2_output" << endl;
    for(int f = 0; f < 5; f++){
        cout << fixp_dense2_output[0][f] << "   ";
    }
    cout << endl;
    // Compute Mean-Squared-Error
    for(int f = 0; f < 1; f++)
    {
        for(int i = 0; i < 32; i++)
        {
            for(int j = 0; j < 8; j++)
            {
                //cout << fixp_conv_layer_output_feature_map1[f][i][j] << "   ";        
            }
            //cout << endl;
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