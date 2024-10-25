#include <iostream>
#include <string>
#include <fstream>
#include "C:\Users\Gabriele\Desktop\VS Work Dir\Progetti C++\Librerie\MLP\Eigen\Eigen"
#include "C:\Users\Gabriele\Desktop\VS Work Dir\Progetti C++\Librerie\MLP\Eigen\Dense"
#include "..\Headers\Network.h"

using namespace std;
using namespace Eigen;

void load_vals (string pi, VectorXd& pixels)
{
    int i = 0;
    for(auto s = pi.begin() + 1, p = pi.begin(); s != pi.end(); s++)
    {
        if(*s == ',')
        {
            double t = stod(std::string(p, s));
            pixels(i) = t;
            p = s + 1;
            i++;
        }
    }
}

template <typename T>
void load (ifstream& file, vector<T>& pixels, const vector<int>& dim, int in)
{
    string s;
    for(long long unsigned int i = 0; i < dim.size() - 1; i++)
    {
        VectorXd tmp = VectorXd::Zero(dim[i]);
        MatrixXd m;
        if(in == 0)
            m = MatrixXd::Zero(dim.at(i + 1), dim.at(i));
        else
            m = VectorXd::Zero(dim.at(i + 1));
        for(int r = 0; r < m.rows(); r++)
        {
            file >> s;
            if(s == "**")
                file >> s;
            load_vals(s, tmp);
            for(int c = 0; c < m.cols(); c++)
            {

                m(r, c) = tmp(c);
            }
        }
        pixels.push_back(m);
    }
}


int main ()
{   
    ifstream f_data ("..\\csv_files\\mnist_test.csv");
    vector<string> data;
    while(f_data.peek() != EOF) 
    {
        string tmp;
        f_data >> tmp;
        data.push_back(tmp);
    }
    ifstream w ("..\\csv_files\\trained_weights.csv");
    ifstream b ("..\\csv_files\\trained_biases.csv");
    ifstream t_data ("..\\digit.csv");
    vector<MatrixXd> weights = vector<MatrixXd>();
    vector<VectorXd> biases = vector<VectorXd>();
    vector<int> dim {784, 200, 10};
    if(w.good())
        load(w, weights, dim, 0);
    if(b.good())
        load(b, biases, dim, 1);
    Network n = Network(weights, biases, dim);
    int succ = 0;
/*     for(long long unsigned int j = 0; j < data.size(); j++)
    {
        string x = data.at(j);
        int cifra = stoi(string(x.begin(), x.begin() + 1));
        n.feed_forward(string(x.begin() + 2, x.end()));
        for(int i = 0; i < 10; i++)
        {
            if((round(n.getNeurons(2)(i)) == 1) && i == cifra)
                succ++;
        }
        if(j % 1000 == 0)
            cout << "Data " << j << ", succes rate: " << (succ / 10000.) * 100 << "\n\n";

    }
    cout << "Final succes rate: " << (succ / 10000.) * 100; */
    string s;
    t_data >> s;
    n.feed_forward(s);
    int cifra = 0;
    for(int i = 0; i < 10; i++)
    {
        if(round(n.getNeurons(2)(i)) == 1)
            cifra = i;
        cout << n.getNeurons(2)(i) << endl;
    }
    ofstream r ("..\\result.txt");
    r.write(to_string(cifra).c_str(), 1);
    return 0;
}
