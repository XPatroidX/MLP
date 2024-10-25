#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include <cmath>
#include <iostream>
#include <fstream>
#include "C:\Users\Gabriele\Desktop\VS Work Dir\Progetti C++\Librerie\MLP\Eigen\Eigen"
#include "C:\Users\Gabriele\Desktop\VS Work Dir\Progetti C++\Librerie\MLP\Eigen\Dense"
#include <string>
#include <vector>
#include <algorithm>
#include <random>

using std::vector;
using std::string;
using Eigen::VectorXd;
using Eigen::MatrixXd;

class Network
{
    public:
        Network(vector<string> _data, double _l_rate, vector<int> dim);
        Network(vector<MatrixXd>& weigths, vector<VectorXd>& biases, vector<int>& dim);

        VectorXd output () {return neurons[layers - 1];};
        const MatrixXd& getWeights(int i) {return weights[i];};
        const VectorXd& getBiases(int i) {return biases[i];};
        const VectorXd& getNeurons(int i) {return neurons[i];};
        const VectorXd& getZ(int i) {return z[i];};
        void learn (int epoch, int mini_batch);
        void feed_forward(string sample);


    private:
        void exp_values (string sample, VectorXd& e_valuesm);
        double SGD(int mini_batch, unsigned long long int& n_datan_data);
        void SGD_b (vector<VectorXd>& p_d_biases, string sample);
        void SGD_w (vector<MatrixXd>& p_d_weights, const vector<VectorXd>& p_d_biases);
        void step (const vector<MatrixXd>& p_d_weights, const vector<VectorXd>& p_d_biases, int mini_batch, unsigned long long int& n_data);
        double cost(string sample);

        std::default_random_engine rng = std::default_random_engine {};
        vector<string> data;
        double l_rate;
        int layers;
        vector<VectorXd> z;
        vector<MatrixXd> weights;
        vector<VectorXd> biases;
        vector<VectorXd> neurons;
};

#endif