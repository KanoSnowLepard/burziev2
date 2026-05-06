// ==============================================
// Лабораторная работа: структуры данных
// Задача: N серых и M белых мышей по кругу
// Кошка съедает каждую S-ю мышь, начиная с серой
// Найти порядок, если осталось K серых и L белых
// ==============================================
// Автор: Бурзиев Иван Сергеевич
// Группа: РПИа о25
// ==============================================
#include <iostream>
#include <vector>
#include <complex>
#include <chrono>
#include <omp.h>
#include <algorithm>
#include <windows.h>

using namespace std;
using namespace std::chrono;

typedef complex<float> ComplexF;

void multiply_naive(const vector<ComplexF>& A, const vector<ComplexF>& B, vector<ComplexF>& C, int n) {
    fill(C.begin(), C.end(), ComplexF(0, 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            ComplexF sum(0, 0);
            for (int k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}
void multiply_variant2(const vector<ComplexF>& A, const vector<ComplexF>& B, vector<ComplexF>& C, int n) {
    fill(C.begin(), C.end(), ComplexF(0, 0));
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            ComplexF a_val = A[i * n + k];
            for (int j = 0; j < n; ++j) {
                C[i * n + j] += a_val * B[k * n + j];
            }
        }
    }
}

void multiply_optimized(const vector<ComplexF>& A, const vector<ComplexF>& B, vector<ComplexF>& C, int n) {
    fill(C.begin(), C.end(), ComplexF(0, 0));
    const int blockSize = 32; 

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i += blockSize) {
        for (int j = 0; j < n; j += blockSize) {
            for (int k = 0; k < n; k += blockSize) {
                for (int ii = i; ii < min(i + blockSize, n); ++ii) {
                    for (int kk = k; kk < min(k + blockSize, n); ++kk) {
                        ComplexF a_val = A[ii * n + kk];
                        int row_idx = ii * n;
                        int k_idx = kk * n;
                        for (int jj = j; jj < min(j + blockSize, n); ++jj) {
                            C[row_idx + jj] += a_val * B[k_idx + jj];
                        }
                    }
                }
            }
        }
    }
}

void print_stats(const char* name, double seconds, long long ops) {
    double mflops = (ops / seconds) * 1e-6;
    cout << "Вариант: " << name << endl;
    cout << "Время (t): " << seconds << " сек." << endl;
    cout << "Производительность (p): " << mflops << " MFlops" << endl;
    cout << "-----------------------------------" << endl << flush;
}
int main() {
    SetConsoleCP(65001);
    SetConsoleOutputCP(65001);

    const int n = 4096;
    const long long ops = 2LL * n * n * n; 

    vector<ComplexF> A(n * n);
    vector<ComplexF> B(n * n);
    vector<ComplexF> C(n * n);

    for (int i = 0; i < n * n; ++i) {
        A[i] = { (float)(rand() % 10), (float)(rand() % 10) };
        B[i] = { (float)(rand() % 10), (float)(rand() % 10) };
    }

    cout << "Запуск расчетов для матрицы " << n << "x" << n << "..." << endl << endl;

    auto t3s = high_resolution_clock::now();
    multiply_optimized(A, B, C, n);
    auto t3e = high_resolution_clock::now();
    print_stats("3-й вариант (Оптимизированный)", duration<double>(t3e - t3s).count(), ops);

    auto t2s = high_resolution_clock::now();
    multiply_variant2(A, B, C, n);
    auto t2e = high_resolution_clock::now();
    print_stats("2-й вариант (i-k-j оптимизация)", duration<double>(t2e - t2s).count(), ops);

    auto t1s = high_resolution_clock::now();
    multiply_naive(A, B, C, n);
    auto t1e = high_resolution_clock::now();
    print_stats("1-й вариант (Линейная алгебра)", duration<double>(t1e - t1s).count(), ops);

    system("pause");
    return 0;
}