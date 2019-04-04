#include <s2pp.h>
#include "libnux/mailbox.h"
#include "libnux/dls_v2.h"
#include "libnux/random.h"
#include "spikes.h"
#include "Utils.h"
#include "build/config.h"

#define N_ACTIONS_MAX 16
#define N_RUNS_MAX 50
#define N_CYCLES_FOR_INHIBITION 400
#define INITIALIZE_ON_CHIP

// stimulation spikes
#define SPIKE_STIMULATE_ADDRESS 21
#define SPIKE_STIMULATE_ROW 31

#define DEBUG_BASE 0xD00
uint32_t debug_counter = 0;

uint32_t mapping[] = {13, 30};

uint32_t seed = 1234123;

typedef struct _mab_task {
    uint32_t n_pulls;
    uint32_t num_actions;
    uint32_t n_runs;
    uint32_t n_batch;
    uint32_t p_reward[N_ACTIONS_MAX * N_RUNS_MAX];
} mab_task;

#ifdef GREEDY

typedef struct _neural_network {
    uint32_t weights[N_ACTIONS_MAX];
    uint32_t alpha_0;
    uint32_t beta_0;
} neural_network;

#elif ANN

#define N_INPUT 5
#define N_HIDDEN 8

#define INT_MAX 0x7fffffff
#define INT_MIN -INT_MAX
#define UINT_MAX 0xffffffff

int sig[] = {241, 257, 273, 291, 310, 330, 351, 374, 398, 
424, 452, 481, 512, 545, 580, 618, 658, 
700, 746, 794, 845, 900, 958, 1020, 1086, 
1156, 1231, 1310, 1395, 1485, 1581, 1683, 1792, 
1908, 2031, 2163, 2302, 2451, 2610, 2778, 2958, 
3149, 3353, 3569, 3800, 4046, 4307, 4586, 4882, 
5198, 5534, 5891, 6272, 6677, 7109, 7568, 8057, 
8578, 9133, 9723, 10351, 11020, 11732, 12490, 13298, 
14157, 15072, 16046, 17083, 18187, 19362, 20614, 21946, 
23364, 24874, 26482, 28193, 30015, 31955, 34020, 36218, 
38559, 41051, 43703, 46528, 49535, 52736, 56144, 59772, 
63634, 67747, 72125, 76786, 81748, 87030, 92655, 98642, 
105017, 111803, 119028, 126719, 134908, 143626, 152907, 162788, 
173307, 184506, 196429, 209122, 222635, 237021, 252337, 268642, 
286000, 304481, 324155, 345100, 367398, 391136, 416409, 443313, 
471956, 502449, 534911, 569471, 606262, 645430, 687127, 731518, 
778775, 829083, 882641, 939656, 1000353, 1064968, 1133755, 1206983, 
1284937, 1367922, 1456264, 1550307, 1650418, 1756988, 1870434, 1991198, 
2119752, 2256596, 2402265, 2557326, 2722382, 2898078, 3085096, 3284165, 
3496057, 3721598, 3961661, 4217179, 4489143, 4778607, 5086691, 5414588, 
5763565, 6134970, 6530234, 6950883, 7398534, 7874908, 8381835, 8921258, 
9495242, 10105982, 10755806, 11447191, 12182766, 12965319, 13797815, 14683398, 
15625403, 16627371, 17693056, 18826440, 20031744, 21313440, 22676270, 24125253, 
25665706, 27303256, 29043856, 30893803, 32859755, 34948747, 37168207, 39525980, 
42030340, 44690011, 47514188, 50512551, 53695286, 57073104, 60657254, 64459547, 
68492362, 72768671, 77302042, 82106659, 87197322, 92589461, 98299132, 104343019, 
110738432, 117503289, 124656109, 132215986, 140202562, 148635989, 157536888, 166926294, 
176825593, 187256449, 198240719, 209800353, 221957289, 234733325, 248149986, 262228372, 
276988992, 292451586, 308634936, 325556655, 343232976, 361678522, 380906069, 400926304, 
421747581, 443375670, 465813513, 489060985, 513114667, 537967630, 563609241, 590024999, 
617196386, 645100769, 673711332, 702997045, 732922695, 763448954, 794532500, 826126195, 
858179312, 890637815, 923444686, 956540298, 989862833, 1023348727, 1056933148, 1090550498, 
1124134919, 1157620813, 1190943348, 1224038960, 1256845831, 1289304334, 1321357451, 1352951146, 
1384034692, 1414560951, 1444486601, 1473772314, 1502382877, 1530287260, 1557458647, 1583874405, 
1609516016, 1634368979, 1658422661, 1681670133, 1704107976, 1725736065, 1746557342, 1766577577, 
1785805124, 1804250670, 1821926991, 1838848710, 1855032060, 1870494654, 1885255274, 1899333660, 
1912750321, 1925526357, 1937683293, 1949242927, 1960227197, 1970658053, 1980557352, 1989946758, 
1998847657, 2007281084, 2015267660, 2022827537, 2029980357, 2036745214, 2043140627, 2049184514, 
2054894185, 2060286324, 2065376987, 2070181604, 2074714975, 2078991284, 2083024099, 2086826392, 
2090410542, 2093788360, 2096971095, 2099969458, 2102793635, 2105453306, 2107957666, 2110315439, 
2112534899, 2114623891, 2116589843, 2118439790, 2120180390, 2121817940, 2123358393, 2124807376, 
2126170206, 2127451902, 2128657206, 2129790590, 2130856275, 2131858243, 2132800248, 2133685831, 
2134518327, 2135300880, 2136036455, 2136727840, 2137377664, 2137988404, 2138562388, 2139101811, 
2139608738, 2140085112, 2140532763, 2140953412, 2141348676, 2141720081, 2142069058, 2142396955, 
2142705039, 2142994503, 2143266467, 2143521985, 2143762048, 2143987589, 2144199481, 2144398550, 
2144585568, 2144761264, 2144926320, 2145081381, 2145227050, 2145363894, 2145492448, 2145613212, 
2145726658, 2145833228, 2145933339, 2146027382, 2146115724, 2146198709, 2146276663, 2146349891, 
2146418678, 2146483293, 2146543990, 2146601005, 2146654563, 2146704871, 2146752128, 2146796519, 
2146838216, 2146877384, 2146914175, 2146948735, 2146981197, 2147011690, 2147040333, 2147067237, 
2147092510, 2147116248, 2147138546, 2147159491, 2147179165, 2147197646, 2147215004, 2147231309, 
2147246625, 2147261011, 2147274524, 2147287217, 2147299140, 2147310339, 2147320858, 2147330739, 
2147340020, 2147348738, 2147356927, 2147364618, 2147371843, 2147378629, 2147385004, 2147390991, 
2147396616, 2147401898, 2147406860, 2147411521, 2147415899, 2147420012, 2147423874, 2147427502, 
2147430910, 2147434111, 2147437118, 2147439943, 2147442595, 2147445087, 2147447428, 2147449626, 
2147451691, 2147453631, 2147455453, 2147457164, 2147458772, 2147460282, 2147461700, 2147463032, 
2147464284, 2147465459, 2147466563, 2147467600, 2147468574, 2147469489, 2147470348, 2147471156, 
2147471914, 2147472626, 2147473295, 2147473923, 2147474513, 2147475068, 2147475589, 2147476078, 
2147476537, 2147476969, 2147477374, 2147477755, 2147478112, 2147478448, 2147478764, 2147479060, 
2147479339, 2147479600, 2147479846, 2147480077, 2147480293, 2147480497, 2147480688, 2147480868, 
2147481036, 2147481195, 2147481344, 2147481483, 2147481615, 2147481738, 2147481854, 2147481963, 
2147482065, 2147482161, 2147482251, 2147482336, 2147482415, 2147482490, 2147482560, 2147482626, 
2147482688, 2147482746, 2147482801, 2147482852, 2147482900, 2147482946, 2147482988, 2147483028, 
2147483066, 2147483101, 2147483134, 2147483165, 2147483194, 2147483222, 2147483248, 2147483272, 
2147483295, 2147483316, 2147483336, 2147483355, 2147483373, 2147483389, 2147483405, };
#define E_SIGMOID 9
int sigmoid(int x) {
    x = x >> 1;
    x += 1 << 30;
    return sig[x >> (31 - E_SIGMOID)] >> 5;
}

typedef struct _neural_network {
    uint32_t weights[N_ACTIONS_MAX];
    uint32_t learning_rate;
} neural_network;

typedef struct _ann_parametrization {
    int w1[N_INPUT][N_HIDDEN];
    int b1[N_HIDDEN];
    int w2[N_HIDDEN];
} ann_parametrization;

#else

typedef struct _neural_network {
    uint32_t weights[N_ACTIONS_MAX];
    uint32_t learning_rates[N_ACTIONS_MAX];
    uint32_t learning_rate_decay;
    uint32_t learning_rate_base;
    uint32_t weight_prior;
} neural_network;

#endif

typedef struct _mab_stats {
    uint32_t wins[N_ACTIONS_MAX];
    uint32_t plays[N_ACTIONS_MAX];
} mab_stats;

void *_memset(void *start, int value, int num) {
  unsigned char *ptr = (unsigned char *) start;
  unsigned char *ptr2 = ptr;
  while (ptr2 + num > ptr) {
    *ptr = (unsigned char) value;
    ptr ++;
  }
  return (void *) ptr2;
}

uint64_t udiv64(uint64_t n, uint64_t d) {
    uint64_t q = 0;
    while (n >= d) {
        uint32_t i = 0;
        uint64_t d_t = d;
        while (n >= (d_t << 1) && ++i)
            d_t <<= 1;
        q |= 1 << i;
        n -= d_t;
    }
    return q;
}

uint32_t udiv32(uint32_t n, uint32_t d) {
    uint32_t q = 0;
    while (n >= d) {
        uint32_t i = 0, d_t = d;
        while (n >= (d_t << 1) && ++i)
            d_t <<= 1;
        q |= 1 << i;
        n -= d_t;
    }
    return q;
}

void initialize(neural_network *network, mab_stats *stats, uint32_t num_actions) {
#ifdef INITIALIZE_ON_CHIP
    uint32_t i, j;
    uint32_t weight_offset;
    // uint32_t i_bound = num_actions > 16 ? 2 : 1;
    uint32_t i_bound = 2;
    uint8_t weight_vector[32];
    _memset((void *) weight_vector, 0, 32);
    for (i = 0; i < num_actions; i ++) {
        weight_vector[mapping[i]] = (uint8_t) (network->weights[i] >> 26);
    }
    for (i = 0; i < i_bound; i ++) {
        // uint32_t j_bound = num_actions > 16 && i == 0 ? 16 : num_actions;
	uint32_t j_bound = 16;
        register vector uint8_t vec_weights;
        
        weight_offset = dls_num_synapse_vectors - 2 * (32 - SPIKE_STIMULATE_ROW) + i;
        
        // read in synapse weights
        asm volatile (
            "fxvinx %[vec_weights], %[dls_weight_base], %[weight_offset]\n"
            : [vec_weights] "=&kv" (vec_weights)
            : [weight_offset] "r" (weight_offset),
              [dls_weight_base] "b" (dls_weight_base)
            : );
        
        // set new
        for (j = 0; j < j_bound; j ++)
            vec_weights[j] = weight_vector[16 * i + j];
        
        // write back in
        asm volatile (
            "fxvoutx %[vec_weights], %[dls_weight_base], %[weight_offset]\n"
            :
            : [weight_offset] "r" (weight_offset),
              [vec_weights] "kv" (vec_weights),
              [dls_weight_base] "b" (dls_weight_base)
            : );
    }
#endif
    
    _memset((void *) stats->wins, 0, N_ACTIONS_MAX * sizeof(uint32_t));
    _memset((void *) stats->plays, 0, N_ACTIONS_MAX * sizeof(uint32_t));
}

uint32_t get_reward(uint32_t action, mab_task *task, uint32_t run_index) {
    if ((uint32_t)random_lcg(&seed) < task->p_reward[action + task->num_actions * run_index])
        return 1;
    return 0;
}

void load_env(mab_task *task) {
    uint32_t i, j;
    task->n_pulls = *((uint32_t *) &mailbox_base);
    task->num_actions = (uint8_t) (*((uint32_t *) &mailbox_base + 1));
    task->n_runs = *((uint32_t *) &mailbox_base + 2);
    task->n_batch = *((uint32_t *) &mailbox_base + 3);
    for (j = 0; j < task->n_runs; j ++)
        for (i = 0; i < task->num_actions; i ++)
            task->p_reward[j * task->num_actions + i] = *((uint32_t *) &mailbox_base + 4 + j * task->num_actions + i);
}

void reset_network(neural_network *network) {
#ifdef GREEDY
    uint32_t weight_prior = (uint32_t) udiv64((uint64_t) network->alpha_0 << 32, network->alpha_0 + network->beta_0);
#endif
    for (uint32_t i = 0; i < N_ACTIONS_MAX; i ++) {
#if ! ANN && ! GREEDY
        network->learning_rates[i] = network->learning_rate_base;
#endif
#if ANN
        network->weights[i] = 48 << 26;
#elif GREEDY
        network->weights[i] = weight_prior;
#else
        network->weights[i] = network->weight_prior;
#endif
    }
}

#ifdef GREEDY
void load_parameters(neural_network *network, uint32_t num_actions, uint32_t n_runs) {
    uint32_t bandit_probabilities_offset = num_actions * n_runs;
    uint32_t alpha_0 = *((uint32_t *) &mailbox_base + 4 + bandit_probabilities_offset);
    uint32_t beta_0 = *((uint32_t *) &mailbox_base + 5 + bandit_probabilities_offset);
    network->alpha_0 = alpha_0;
    network->beta_0 = beta_0;
    seed = *((int *) &mailbox_base + 6 + bandit_probabilities_offset);
    uint32_t weight_prior = (uint32_t) udiv64((uint64_t) network->alpha_0 << 32, network->alpha_0 + network->beta_0);
    for (uint32_t i = 0; i < N_ACTIONS_MAX; i ++) {
        network->weights[i] = weight_prior;
    }
}
#elif ANN
void load_parameters(neural_network *network, ann_parametrization *ann, uint32_t num_actions, uint32_t n_runs) {
    uint32_t offset = num_actions * n_runs;
    seed = *((int *) &mailbox_base + 4 + offset);
    network->learning_rate = *((uint32_t *) &mailbox_base + 5 + offset);
    for (uint32_t i = 0; i < N_INPUT; i ++) {
	    for (uint32_t j = 0; j < N_HIDDEN; j ++) {
		    ann->w1[i][j] = *((int *) &mailbox_base + 6 + offset + i * N_HIDDEN + j);
	    }
    }
    offset += N_INPUT * N_HIDDEN;
    for (uint32_t i = 0; i < N_HIDDEN; i ++)
	    ann->b1[i] = *((int *) &mailbox_base + 6 + offset + i);
    offset += N_HIDDEN;
    for (uint32_t i = 0; i < N_HIDDEN; i ++)
	    ann->w2[i] = *((int *) &mailbox_base + 6 + offset + i);
    offset += N_HIDDEN;
}
#else
void load_parameters(neural_network *network, uint32_t num_actions, uint32_t n_runs) {
    uint32_t bandit_probabilities_offset = num_actions * n_runs;
    uint32_t learning_rate = *((uint32_t *) &mailbox_base + 4 + bandit_probabilities_offset);
    uint32_t weight_prior = *((uint32_t *) &mailbox_base + 5 + bandit_probabilities_offset);
    seed = *((int *) &mailbox_base + 6 + bandit_probabilities_offset);
    network->learning_rate_decay = *((uint32_t *) &mailbox_base + 7 + bandit_probabilities_offset);
    network->learning_rate_base = learning_rate;
    network->weight_prior = weight_prior;
    for (uint32_t i = 0; i < N_ACTIONS_MAX; i ++) {
        network->learning_rates[i] = learning_rate;
        network->weights[i] = weight_prior;
    }
}
#endif

// assume clear on read of spike counters
// returns != 0 if one spike counter is nonzero
int update_spike_counters(uint32_t *spike_counters, uint32_t num_actions) {
    uint32_t i;
    int nonzero = 0;
    for (i = 0; i < num_actions; i ++) {
        spike_counters[i] += dls_rates_base[mapping[i]];
        if (spike_counters[i])
            nonzero = 1;
    }
    return nonzero;
}

void reset_spike_counters(uint32_t *spike_counters, uint32_t num_actions) {
    _memset(spike_counters, 0, num_actions * sizeof(uint32_t));
}

#ifdef GREEDY
void update_weights_mean(uint32_t action, neural_network *network, mab_stats *stats) {
    // use last synapse driver for state -> action weights
    uint32_t weight_offset = dls_num_synapse_vectors - 2 * (32 - SPIKE_STIMULATE_ROW) + 1;
    register vector uint8_t vec_weights;
    
    if (mapping[action] < 16)
        weight_offset --;
    
    asm volatile (
        "fxvinx %[vec_weights], %[dls_weight_base], %[weight_offset]\n"
        : [vec_weights] "=&kv" (vec_weights)
        : [weight_offset] "r" (weight_offset),
          [dls_weight_base] "b" (dls_weight_base)
        : );
    
    network->weights[action] = (uint32_t) udiv64(((uint64_t) stats->wins[action] + network->alpha_0) << 32, stats->plays[action] + network->alpha_0 + network->beta_0);
    vec_weights[mapping[action] % 16] = (uint8_t) (network->weights[action] >> 26);
    
    asm volatile (
        "fxvoutx %[vec_weights], %[dls_weight_base], %[weight_offset]\n"
        :
        : [weight_offset] "r" (weight_offset),
          [vec_weights] "kv" (vec_weights),
          [dls_weight_base] "b" (dls_weight_base)
        : );
}
#elif ANN
void update_weights_ann(uint32_t action, uint32_t reward, uint32_t step, neural_network *network, ann_parametrization *ann) {
    // use last synapse driver for state -> action weights
    uint32_t weight_offset = dls_num_synapse_vectors - 2 * (32 - SPIKE_STIMULATE_ROW) + 1;
    int ann_input[N_INPUT];
    long long int t_hidden;
    long long int output;
    register vector uint8_t vec_weights;
    
    if (mapping[action] < 16)
        weight_offset --;
    
    asm volatile (
        "fxvinx %[vec_weights], %[dls_weight_base], %[weight_offset]\n"
        : [vec_weights] "=&kv" (vec_weights)
        : [weight_offset] "r" (weight_offset),
          [dls_weight_base] "b" (dls_weight_base)
        : );
    
    for (uint32_t a = 0; a < 2; a ++) {
        //ann_input[0] = step << 20;
	int flag_value = 1 << 26;
        ann_input[0] = step << 18;
        ann_input[1] = reward ? flag_value : -flag_value;
        ann_input[2] = action == a ? flag_value : -flag_value;
        ann_input[3] = (network->weights[a] >> 6);
        ann_input[4] = (network->weights[1 - a] >> 6);

        // matmul
        output = 0;
        for (uint32_t i = 0; i < N_HIDDEN; i ++) {
            t_hidden = ann->b1[i];
            for (uint32_t j = 0; j < N_INPUT; j ++) {
                t_hidden += ((long long int) ann_input[j] * ann->w1[j][i]) >> 26;
            }
            // saturate
            if (t_hidden < (long long int) INT_MIN)
                t_hidden = INT_MIN;
            else if (t_hidden >= (long long int) INT_MAX)
                t_hidden = INT_MAX;

            // activate
            t_hidden = sigmoid((int) t_hidden);
            output += (t_hidden * ann->w2[i]) >> 26;
        }
        // saturate
        if (output < (long long int) INT_MIN)
            output = INT_MIN;
        else if (output >= (long long int) INT_MAX)
            output = INT_MAX;
        output = (output * network->learning_rate) >> 32;

        if (output < 0) {
            output *= -1;
            uint32_t temp = network->weights[a] - (uint32_t) output;
            if (temp > network->weights[a])
                network->weights[a] = 0;
            else
                network->weights[a] = temp;
        } else {
            uint32_t temp = network->weights[a] + (uint32_t) output;
            if (temp < network->weights[a])
                network->weights[a] = UINT_MAX;
            else
                network->weights[a] = temp;
        }

	//uint32_t temp = network->weights[a] >> 26;
	//mailbox_write(0x400 + 2 * step + a, (void *) ((char *) &temp + 3), 1);
	uint32_t temp = network->weights[a] >> 27;
	temp += 32;

        //vec_weights[mapping[a] % 16] = (uint8_t) (network->weights[a] >> 26);
	vec_weights[mapping[a] % 16] = (uint8_t) temp;
    }
    
    asm volatile (
        "fxvoutx %[vec_weights], %[dls_weight_base], %[weight_offset]\n"
        :
        : [weight_offset] "r" (weight_offset),
          [vec_weights] "kv" (vec_weights),
          [dls_weight_base] "b" (dls_weight_base)
        : );
}
#else
void update_weights_local(uint32_t action, uint32_t reward, neural_network *network) {
    // use last synapse driver for state -> action weights
    uint32_t weight_offset = dls_num_synapse_vectors - 2 * (32 - SPIKE_STIMULATE_ROW) + 1;
    uint32_t learning_rate = network->learning_rates[action];
    uint32_t learning_rate_decay = network->learning_rate_decay;
    uint64_t weight = network->weights[action];
    register vector uint8_t vec_weights;
    
    if (mapping[action] < 16)
        weight_offset --;
    
    asm volatile (
        "fxvinx %[vec_weights], %[dls_weight_base], %[weight_offset]\n"
        : [vec_weights] "=&kv" (vec_weights)
        : [weight_offset] "r" (weight_offset),
          [dls_weight_base] "b" (dls_weight_base)
        : );
    
    weight = weight * ((~learning_rate + 1) & (((uint64_t) 1 << 32) - 1));
    weight = weight >> 32;
    
    if (reward)
        weight += learning_rate;
    network->weights[action] = (uint32_t) weight;
    vec_weights[mapping[action] % 16] = (uint8_t) ((network->weights[action] >> 26) + 0);

    if (learning_rate_decay != 0) {
        uint64_t temp_learning_rate = learning_rate;
        temp_learning_rate = temp_learning_rate * learning_rate_decay;
        temp_learning_rate = temp_learning_rate >> 32;
        network->learning_rates[action] = (uint32_t) temp_learning_rate;
    }
    
    asm volatile (
        "fxvoutx %[vec_weights], %[dls_weight_base], %[weight_offset]\n"
        :
        : [weight_offset] "r" (weight_offset),
          [vec_weights] "kv" (vec_weights),
          [dls_weight_base] "b" (dls_weight_base)
        : );
}
#endif

void set_weight(uint8_t row, uint8_t column, uint8_t weight) {
    uint32_t weight_offset = row * 2 + column / 16;
    register vector uint8_t vec_weights;
    
    asm volatile (
        "fxvinx %[vec_weights], %[dls_weight_base], %[weight_offset]\n"
        : [vec_weights] "=&kv" (vec_weights)
        : [weight_offset] "r" (weight_offset),
          [dls_weight_base] "b" (dls_weight_base)
        : );
    
    vec_weights[column % 16] = weight;
    
    asm volatile (
        "fxvoutx %[vec_weights], %[dls_weight_base], %[weight_offset]\n"
        :
        : [weight_offset] "r" (weight_offset),
          [vec_weights] "kv" (vec_weights),
          [dls_weight_base] "b" (dls_weight_base)
        : );
}

void uitoa(uint32_t number, char *buffer) { // base = 10
    uint32_t base = 10;
    uint32_t i;
    for (i = 0; number > 0; i ++) {
        buffer[i] = '0' + (number % base);
        number = udiv32(number, base);
    }
    buffer[i] = 0;
}

void debug_message(char *msg) {
    while (*msg)
        mailbox_write(DEBUG_BASE + debug_counter++, (void *) msg++, 1);
}

uint32_t get_action(uint32_t *spike_counters, uint32_t num_actions, uint32_t pull) {
    uint32_t nonzero = 0;
    uint32_t randint;
    uint32_t num_nonzero = 0;
    uint32_t active_counter = 0;
    uint32_t action = 0;
    uint32_t i;
    
    // define spike packet
    spike_t spike;
    spike.addr = SPIKE_STIMULATE_ADDRESS;
    spike.row_mask = 1 << SPIKE_STIMULATE_ROW;
    set_weight(SPIKE_STIMULATE_ROW, SPIKE_STIMULATE_ROW, 63);
    
    // check if triggering neuron did not stop spiking:
    if (dls_rates_base[SPIKE_STIMULATE_ROW] > 0) {
        uint32_t fail = 0xdeaddead;
        mailbox_write(0xff4, (void *) &fail, 4);
    }
    reset_spike_counters(spike_counters, num_actions);
    // loop until action spikes arrived
//     i = 0;
//     while (! nonzero && i < 20) {
//         spikes_send(&spike);
//         wait_cycles(200);
//         nonzero = update_spike_counters(spike_counters, num_actions);
//         i ++;
//     }
    // activate action triggering neuron by its own recurrence it will continue
    // to spike forever until inhibited from action neurons
    spikes_send(&spike);
    wait100();
    // stop state neuron spiking
    nonzero = 0;
    for (i = 0; i < 10 && (! nonzero); i ++) {
        nonzero = update_spike_counters(spike_counters, num_actions);
        // wait100();
    }
    
    set_weight(SPIKE_STIMULATE_ROW, SPIKE_STIMULATE_ROW, 0);
    for (i = 0; i < 6; i ++)
        wait100();
    // reset triggering neuron spike counter
    i = dls_rates_base[SPIKE_STIMULATE_ROW];
    // uint32_t temp = spike_counters[0] | 0x80;
    // mailbox_write(0x400 + pull, (void *) ((char *) &temp + 3), 1);
    // temp = spike_counters[1] | 0x80;
    // mailbox_write(0x500 + pull, (void *) ((char *) &temp + 3), 1);
    // mailbox_write(0x400 + pull, (void *) &spike_counters[0] + 3, 1);
    // mailbox_write(0x500 + pull, (void *) &spike_counters[1] + 3, 1);
    
    // nonzero = update_spike_counters(spike_counters, num_actions);
    if (! nonzero) {
        uint32_t fail = 0xdeadbeef;
        mailbox_write(0xff0, (void *) &fail, 4);
	//return 0;
        return (((uint32_t) random_lcg(&seed)) >> 24) % num_actions;
    }
    // wait_cycles(N_CYCLES_FOR_INHIBITION);
    // update_spike_counters(spike_counters, num_actions);
    
    for (i = 0; i < num_actions; i ++)
        if (spike_counters[i])
            num_nonzero ++;
    
    // choose randomly among actions
    randint = (uint32_t) random_lcg(&seed) >> 24;
    randint = randint % num_nonzero;
    for (i = 0; i < num_actions; i ++)
        if (spike_counters[i] && active_counter++ == randint) {
            action = i;
            break;
        }
    return action;
}

void start(void)
{
    uint32_t i, run_index, batch_index;
    uint32_t action, reward;
    uint32_t spike_counters[N_ACTIONS_MAX];
    uint8_t to_mailbox;
    char buffer[32];
    
    mab_task task;
    neural_network network;
    mab_stats stats;
#ifdef ANN
    ann_parametrization ann;
#endif
    
    load_env(&task);
#ifndef ANN
    load_parameters(&network, task.num_actions, task.n_runs);
#else
    load_parameters(&network, &ann, task.num_actions, task.n_runs);
#endif
    
    debug_message("n_runs: ");
    uitoa(task.n_runs, buffer);
    debug_message(buffer);
    debug_message("\n");
    
    debug_message("n_batch: ");
    uitoa(task.n_batch, buffer);
    debug_message(buffer);
    debug_message("\n");
    
    for (run_index = 0; run_index < task.n_runs; run_index ++) {
        for (batch_index = 0; batch_index < task.n_batch; batch_index ++) {
            reset_spike_counters(spike_counters, task.num_actions);
            reset_network(&network);
            initialize(&network, &stats, task.num_actions);

            for (i = 0; i < task.n_pulls; i ++) {
                action = get_action(spike_counters, task.num_actions, i);
                reward = get_reward(action, &task, run_index);

                // ________________________________________________________________________________________
                // write action, reward to mailbox
                to_mailbox = reward ? 1 << 7 : 0;
                to_mailbox |= (uint8_t) action;
                mailbox_write((run_index * task.n_batch + batch_index) * task.n_pulls + i, &to_mailbox, 1);
                // ----------------------------------------------------------------------------------------

                stats.plays[action] ++;
                stats.wins[action] += reward;

                // ________________________________________________________________________________________
                // synaptic update
#ifdef GREEDY
                update_weights_mean(action, &network, &stats);
#elif ANN
                update_weights_ann(action, reward, i, &network, &ann);
#else
                update_weights_local(action, reward, &network);
#endif
                // ----------------------------------------------------------------------------------------
                wait_cycles(1000);
            }
            wait_cycles(1000);
        }
    }
}
