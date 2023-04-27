#include <iostream>
#include <fstream>
#include <cmath>
#include "conv.h" 

using namespace std;

//renamed input_feature_maps to output_feature_maps because the bin is actually the output of the layer
float output_feature_map_map1_golden[1][32][187];
float conv_layer_weights1[32][1][5];
float conv_layer_bias1[32];

float conv_layer_weights2[32][32][5];
float conv_layer_bias2[32];

float conv_layer_weights3[32][32][5];
float conv_layer_bias3[32];

float conv_layer_weights4[32][32][5];
float conv_layer_bias4[32];

fm_t fixp_conv_layer_input_feature_map1[1][1][187];

// fm_t fixp_output_feature_map_map1[1][32][187];
// fm_t fixp_output_feature_map_map1_max[1][32][92];
// fm_t fixp_output_feature_map_map2[1][32][92];
// fm_t fixp_output_feature_map_map2_max[1][32][44];
// fm_t fixp_output_feature_map_map3[1][32][44];
// fm_t fixp_output_feature_map_map4[1][32][44];
// fm_t fixp_output_feature_map_map3_max[1][32][20];
// fm_t fixp_output_feature_map_map4_max[1][32][8];
// fm_t fixp_output_feature_map_map5_max[1][32][2];
// fm_t fixp_dense1_output[1][32];
fm_t output_feature_map[1][5];

//fm_t fixp_output_feature_map_flat[1][64];

wt_t fixp_conv_layer_weights1[32][1][5];
wt_t fixp_conv_layer_bias1[32];
wt_t fixp_conv_layer_weights2[32][32][5];
wt_t fixp_conv_layer_bias2[32];
wt_t fixp_conv_layer_weights3[32][32][5];
wt_t fixp_conv_layer_bias3[32];
wt_t fixp_conv_layer_weights4[32][32][5];
wt_t fixp_conv_layer_bias4[32];

// declare weights and biases for dense1 and dense2
wt_t fixp_dense1_weights[32][64]; 
wt_t fixp_dense1_bias[32];
wt_t fixp_dense2_weights[5][32];
wt_t fixp_dense2_bias[5];


void read_bin_files()
{    
    // Weights 1
    ifstream ifs_conv1("bin/conv1.bin", ios::in | ios::binary);
    ifs_conv1.read((char*)(**fixp_conv_layer_input_feature_map1), 187*1*1*sizeof(float));
    ifs_conv1.close();
    

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
    ifs_conv2_weights.read((char*)(**conv_layer_weights2), 32*32*5*sizeof(float));
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
    ifs_conv3_weights.read((char*)(**conv_layer_weights3), 32*32*5*sizeof(float));
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
    ifs_conv4_weights.read((char*)(**conv_layer_weights4), 32*32*5*sizeof(float));
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

    //fm_t fixp_conv_layer_input_feature_map1[1][1][187] = {1}; //sets input features all to 1


    tiled_conv (
        fixp_conv_layer_input_feature_map1,
        fixp_conv_layer_weights1,
        fixp_conv_layer_bias1,
        fixp_conv_layer_weights2,
        fixp_conv_layer_bias2,
        fixp_conv_layer_weights3,
        fixp_conv_layer_bias3,
        fixp_conv_layer_weights4,
        fixp_conv_layer_bias4,
        fixp_dense1_weights, 
        fixp_dense1_bias,
        fixp_dense2_weights,
        fixp_dense2_bias,
        output_feature_map
    );

    //need to do linear/dense layers here

    std::cout << "C model complete!\n" << std::endl;
    cout << "output_feature_map" << endl;
    for(int f = 0; f < 5; f++){
        cout << output_feature_map[0][f] << "   ";
    }
    cout << endl;
    // Compute Mean-Squared-Error
    for(int f = 0; f < 1; f++)
    {
        for(int i = 0; i < 32; i++)
        {
            for(int j = 0; j < 8; j++)
            {
                //cout << fixp_output_feature_map_map1[f][i][j] << "   ";        
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
            cout << "Actual: " << output_feature_map_map[f][row][col];
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