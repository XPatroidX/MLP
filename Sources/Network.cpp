#include "..\Headers\Network.h"
#include <cmath>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include <algorithm>

using std::vector;
using std::string;
using Eigen::VectorXd;
using Eigen::MatrixXd;

double sigmoide (double x);
double sigmoide_derivative(double x);
VectorXd sigmoide_derivative(VectorXd vec);
void load_val (string pi, VectorXd& pixels);
//std::ofstream _log ("log_d.txt");

/* template <typename T>
void print(T& mat)
{
    for (int r = 0; r < mat.rows(); r++)
    {
        for (int c = 0; c < mat.cols(); c++)
        {
            _log << mat(r, c) << ",";
        }
        _log << std::endl;
    }
} */

Network::Network(vector<MatrixXd>& _weights, vector<VectorXd>& _biases, vector<int>& dim) : weights{_weights}, biases{_biases}
{
    layers = weights.size() + 1;
    for(int i = 0; i < layers - 1; i++)
    {
        VectorXd _z (dim[i + 1]);
        z.push_back(_z);
        VectorXd n (dim[i]);
        neurons.push_back(n);
    }
    VectorXd n_f (dim[layers - 1]);
    neurons.push_back(n_f);
};

Network::Network(vector<string> _data, double _l_rate, vector<int> dim) : data{_data}, l_rate{_l_rate}
{
    layers = dim.size();
    std::srand(time(0));
    for (int i = 0; i < layers - 1; i++)
    {
        MatrixXd m = MatrixXd::Random(dim[i + 1], dim[i]);
        weights.push_back(m);
        VectorXd b = VectorXd::Random(dim[i + 1]);
        biases.push_back(b);
        VectorXd _z (dim[i + 1]);
        z.push_back(_z);
        VectorXd n (dim[i]);
        neurons.push_back(n);
    }
    VectorXd n_f (dim[layers - 1]);
    neurons.push_back(n_f);
}

void Network::learn(int epoch, int mini_batch)
{

    for(int e = 0; e < epoch; e++)
    {
        //_log << "\n\n\n\nEpoch: " << e << std::endl;
        std::cout << "Epoch: " << e + 1;
        double e_cost = 0;
        shuffle(begin(data), end(data), rng);
        for(unsigned long long int n = 0; n < data.size();)
        {
            e_cost += SGD(mini_batch, n);
        }
        std::cout << "\n\ncost at the end of this epoch: " << e_cost / ((double) data.size() / mini_batch) << "\n\n";
    }
}


double Network::SGD (int mini_batch, unsigned long long int& n_data)
{
    vector<VectorXd> p_d_biases = vector<VectorXd>();
    vector<MatrixXd> p_d_weights = vector<MatrixXd>();
    double b_cost = 0;
    for(int m = 0; m < mini_batch && n_data != data.size(); m++, n_data++)
    {
        string d = data.at(n_data);
        feed_forward(string(d.begin() + 2, d.end()));
        SGD_b(p_d_biases, data.at(n_data));
        SGD_w(p_d_weights, p_d_biases);
        b_cost += cost(data.at(n_data));
    }
    step(p_d_weights, p_d_biases, mini_batch, n_data);
    return b_cost / mini_batch;
}

double Network::cost(string sample)
{
    VectorXd e_values = VectorXd::Zero(neurons[layers - 1].size());
    exp_values(sample, e_values);
    double s_cost = 0;
    for(int i = 0; i < neurons[layers - 1].size(); i++)
    {
        if(e_values[i] == 1)
        {
            s_cost += std::pow(neurons.at(layers - 1)(i) - 1, 2);
        }
        else
        {
            s_cost += std::pow(neurons.at(layers - 1)(i), 2);
        }
    }
    return s_cost;
}

void Network::step(const vector<MatrixXd>& p_d_weights, const vector<VectorXd>& p_d_biases, int mini_batch, unsigned long long int& n_data)
{
    for(int l = layers - 2, n = 0; l >= 0; l--, n++)
    {
        VectorXd b_tmp = VectorXd::Zero(biases.at(l).rows());
        MatrixXd w_tmp = MatrixXd::Zero(weights.at(l).rows(), weights.at(l).cols());
        for(int i = 0; i < mini_batch && n_data != data.size(); i++)
        {
            b_tmp += p_d_biases.at((i * (layers - 1)) + n);
            w_tmp += p_d_weights.at((i * (layers - 1)) + n);
        }
        MatrixXd x = l_rate * (w_tmp / mini_batch);
        VectorXd x1 = l_rate * (b_tmp / mini_batch);
        biases.at(l) -= x1;
        weights.at(l) -= x;
    }
}

void Network::SGD_w (vector<MatrixXd>& p_d_weights, const vector<VectorXd>& p_d_biases)
{
    for(int l = layers - 2; l >= 0; l--)
    {
        p_d_weights.push_back(p_d_biases.at(p_d_biases.size() - (l + 1)) * neurons.at(l).transpose());
    }
}

void Network::SGD_b (vector<VectorXd>& p_d_biases, string sample)
{
    VectorXd e_values;
    exp_values(sample, e_values);
    for(int l = layers - 1; l > 0; l--)
    {
        if(l == (layers - 1))
        {
            p_d_biases.push_back((2*(neurons.at(l) - e_values)).cwiseProduct(sigmoide_derivative(z.at(l-1))));
        }
        else
        {
            VectorXd b =(weights.at(l).transpose() * p_d_biases.at(p_d_biases.size() - 1)).cwiseProduct(sigmoide_derivative(z.at(l - 1)));
            p_d_biases.push_back(b);
        }
    }
}

void Network::feed_forward(string sample)
{
    load_val(sample, neurons[0]);
    for(int l = 0; l < layers - 1; l++)
    {
        z.at(l) = weights.at(l) * neurons.at(l) + biases.at(l);
        for (int i = 0; i < biases.at(l).size(); i++)
        {
            neurons.at(l + 1)(i) = sigmoide(z.at(l)(i));
        }
    }
}

void Network::exp_values (string sample, VectorXd& e_values)
{
    e_values.resize(neurons[layers - 1].size());
    short digit = std::stoi(string(sample.begin(), sample.begin() + 1));
    for(int i = 0;  i < e_values.size(); i++)
    {
        if(i - digit == 0)
        {
            e_values(i) = 1;
        }
        else
        {
            e_values(i) = 0;
        }
    }
}

void load_val (string pi, VectorXd& pixels)
{
    int i = 0;
    for(auto s = pi.begin() + 1, p = pi.begin(); s != pi.end(); s++)
    {
        if(*s == ',')
        {
            double t = std::stoi(std::string(p, s)) / 255.;
            pixels(i) = t;
            p = s + 1;
            i++;
        }
        if(s == (pi.end() - 1))
        {
            double t = std::stoi(std::string(s, pi.end())) / 255.;
            pixels(i) = t;
        }
    }
}

VectorXd sigmoide_derivative(VectorXd vec)
{
    VectorXd result;
    result.resize(vec.size());
    for(int r = 0; r < vec.rows(); r++)
    {
        result(r) = sigmoide_derivative(vec(r));
    }
    return result;
}

double sigmoide_derivative(double x)
{
    double tmp = std::exp(x) / std::pow(1 + std::exp(x), 2);
    if(std::isnan(tmp))
    {
            return 0.;
    }
    return tmp;
}

double sigmoide (double x)
{
    double tmp = 1. / (1 + (1. / std::exp(x)));
    if(std::isnan(tmp))
    {
        if(x >= 0)
            return 1.;
        else
            return 0.;
    }
    return tmp;
}

