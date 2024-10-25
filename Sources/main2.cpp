#include <iostream>
#include "..\Headers\Network.h"

using namespace std;

template <typename T>
void print(T& mat, ofstream& file)
{
    for (int r = 0; r < mat.rows(); r++)
    {
        for (int c = 0; c < mat.cols(); c++)
        {
            file << mat(r, c) << ",";
        }
        file << endl;
    }
    file << "**\n";
}

int main()
{
    ifstream f_data ("..\\csv_files\\mnist_train.csv");
    vector<string> data;
    if(f_data.good())
    {
        while(!(f_data.eof())) 
        {
            string tmp;
            f_data >> tmp;
            data.push_back(tmp);
        }
        vector<int> dim {784, 200, 10};
        Network n (data, 3, dim);
        n.learn(50, 10);
        ofstream w ("..\\csv_files\\trained_weights.csv");
        ofstream b ("..\\csv_files\\trained_biases.csv");
        for(long long unsigned int i = 0; i < dim.size(); i++)
        {
            print(n.getWeights(i), w);
            print(n.getBiases(i), b);
        }
    }
    return 0;
}