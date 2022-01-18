#include <cmath>
#include <cstdio>
#include <ctime>
#include <array>
#include <fstream>

#define iterations 200000
#define max_index 120
#define min_index -30
#define base 1.05
#define recall_cost 3.0
#define forget_cost 15.0
#define d_limit 20
#define d_offset 2

using namespace std;

float cal_start_halflife(int difficulty) {
    return 5.25 * pow(difficulty, -0.866);
}

float cal_next_recall_halflife(float h, float p, int d, int recall) {
    if (recall == 1) {
        return exp(1.83) * pow(d, -0.305) * pow(h, 0.765) * exp(1.26 * (1 - p));
    } else {
        return exp(0.5) * pow(d, -0.068)  * pow(h, 0.4) * exp(-0.688 * (1 - p));
    }
}

int cal_halflife_index(float h) {
    return (int) round(log(h) / log(base)) - min_index;
}

float cal_index_halflife(int index) {
    return exp((index + min_index) * log(base));
}

int main() {
    auto halflife_list = new float[max_index - min_index];
    for (int i = 0; i < max_index - min_index; i++) {
        halflife_list[i] = pow(base, i + min_index);
    }
    int index_len = max_index - min_index;
    auto cost_list = new float[d_limit][max_index - min_index];
    for (int d = 1; d <= d_limit; d++) {
        for (int i = 0; i < index_len - 1; i++) {
            cost_list[d - 1][i] = (float) 20000;
        }
        cost_list[d - 1][index_len - 1] = 0;
    }
    auto used_interval_list = new int[d_limit][max_index - min_index];
    auto recall_list = new float[d_limit][max_index - min_index];
    auto next_index = new int[d_limit][max_index - min_index];
    int start_time = (int) time((time_t *) nullptr);
    for (int d = d_limit; d >= 1; d--) {
        float h0 = cal_start_halflife(d);
        int h0_index = cal_halflife_index(h0);
        for (int i = 0; i < iterations; ++i) {
            float h0_cost = cost_list[d - 1][h0_index];
            for (int h_index = index_len - 2; h_index >= 0; h_index--) {
                float halflife = halflife_list[h_index];

                int interval_min;
                int interval_max;

                interval_min = max(1, (int) round(halflife * log(0.95) / log(0.5)));
                interval_max = max(1, (int) round(halflife * log(0.3) / log(0.5)));

                for (int interval = interval_max; interval >= interval_min; interval--) {
                    float p_recall = exp2(-interval / halflife);
                    float recall_h = cal_next_recall_halflife(halflife, p_recall, d, 1);
                    float forget_h = cal_next_recall_halflife(halflife, p_recall, d, 0);
                    int recall_h_index = min(cal_halflife_index(recall_h), index_len - 1);
                    int forget_h_index = max(cal_halflife_index(forget_h), 0);
                    float exp_cost =
                            p_recall * (cost_list[d - 1][recall_h_index] + recall_cost) +
                            (1.0 - p_recall) *
                            (cost_list[min(d - 1 + d_offset, d_limit - 1)][forget_h_index] + forget_cost);
                    if (exp_cost < cost_list[d - 1][h_index]) {
                        cost_list[d - 1][h_index] = exp_cost;
                        used_interval_list[d - 1][h_index] = interval;
                        recall_list[d - 1][h_index] = p_recall;
                        next_index[d - 1][h_index] = recall_h_index;
                    }
                }
            }

            float diff = h0_cost - cost_list[d - 1][h0_index];
            if (i % 1000 == 0) {
                char name[40];
                sprintf(name, "./result/ivl-%d.csv", d);
                ofstream used_interval_out(name);
                sprintf(name, "./result/cost-%d.csv", d);
                ofstream cost_out(name);
                sprintf(name, "./result/recall-%d.csv", d);
                ofstream recall_out(name);
                for (int k = 0; k <= index_len - 1; k++) {
                    used_interval_out << halflife_list[k] << ',' << used_interval_list[d - 1][k] << '\n';
                    cost_out << halflife_list[k] << ',' << cost_list[d - 1][k] << '\n';
                    recall_out << halflife_list[k] << ',' << recall_list[d - 1][k] << '\n';
                }

                int h_index = h0_index;
                int ivl = used_interval_list[d - 1][h_index];
                float cost = cost_list[d - 1][h_index];
                float recall = recall_list[d - 1][h_index];
                do {
                    float h = cal_index_halflife(h_index);
                    printf("h:%10.4f\tivl:%5d\tr:%.4f\tcost:%10.4f\n", h, ivl, recall, cost);
                    h_index = next_index[d - 1][h_index];
                    if (ivl <= 0) break;
                    ivl = used_interval_list[d - 1][h_index];
                    recall = recall_list[d - 1][h_index];
                    cost = cost_list[d - 1][h_index];
                } while (h_index < index_len);
                printf("D %d\titer %d\tdiff %f\ttime %ds\tcost %f\n", d, i, diff,
                       (int) time((time_t *) nullptr) - start_time,
                       cost_list[d - 1][h0_index]);
                if (diff < 0.1 && i > 500) {
                    break;
                }
            }
        }
    }
    return 0;
}