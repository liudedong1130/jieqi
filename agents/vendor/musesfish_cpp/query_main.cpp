// GPL v3 source adapted from miaosiSari/Jieqi.
// Small single-position query wrapper for jieqi-rl.

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

#include "board/aiboard5.h"
#include "score/score.h"

extern bool read_score_table(const char* score_file, short pst[][256]);
extern void IntializeL1();
extern short pstglobal[5][123][256];

namespace {

constexpr int MAX_STATE = 257;
constexpr const char* MOVE_PREFIX = "__MOVE__ ";

struct Query {
    int turn_int = 1;
    int round = 1;
    short score = 0;
    char state[MAX_STATE];
    unsigned char di[VERSION_MAX][2][123];
};

void initialize_engine(const char* score_file) {
    static bool initialized = false;
    if (initialized) return;
    IntializeL1();
    std::memset(pstglobal, 0, sizeof(pstglobal));
    assert(read_score_table(score_file, pstglobal[2]));
    assert(read_score_table(score_file, pstglobal[3]));
    assert(read_score_table(score_file, pstglobal[4]));
    initialized = true;
}

bool read_query(Query& query) {
    if (!(std::cin >> query.turn_int >> query.round >> query.score)) {
        return false;
    }

    std::unordered_map<char, int> counts;
    for (char c : std::string("RNBACP")) {
        int red_count = 0;
        int black_count = 0;
        if (!(std::cin >> red_count >> black_count)) return false;
        counts[c] = red_count;
        counts[static_cast<char>(std::tolower(c))] = black_count;
    }

    std::string dummy;
    std::getline(std::cin, dummy);

    std::memset(query.state, 0, sizeof(query.state));
    int offset = 0;
    for (int row = 0; row < 16; ++row) {
        std::string line;
        if (!std::getline(std::cin, line)) return false;
        if (line.size() < 16) line.resize(16, ' ');
        for (int col = 0; col < 16; ++col) {
            query.state[offset++] = line[col];
        }
    }

    std::memset(query.di, 0, sizeof(query.di));
    for (char c : std::string("RNBACP")) {
        query.di[0][1][static_cast<int>(c)] = static_cast<unsigned char>(counts[c]);
        char lower = static_cast<char>(std::tolower(c));
        query.di[0][0][static_cast<int>(lower)] = static_cast<unsigned char>(counts[lower]);
    }
    return true;
}

std::string solve_query(const Query& query) {
    std::unordered_map<std::string, bool> hist;
    board::AIBoard5 ai(query.state, query.turn_int != 0, query.round, query.di, query.score, &hist);
    return ai.Think();
}

}  // namespace

int main(int argc, char** argv) {
    const char* score_file = argc > 1 ? argv[1] : "score.conf";
    if (argc > 2) {
        musesfish_query_min_depth = std::max(1, std::atoi(argv[2]));
    }
    if (argc > 3) {
        musesfish_query_max_depth = std::max(musesfish_query_min_depth, std::atoi(argv[3]));
    }
    bool loop = argc > 4 && std::string(argv[4]) == "--loop";
    initialize_engine(score_file);

    if (loop) {
        Query query;
        while (read_query(query)) {
            std::string move = solve_query(query);
            std::cout << MOVE_PREFIX << move << std::endl;
        }
        return 0;
    }

    Query query;
    if (!read_query(query)) return 2;
    std::string move = solve_query(query);
    std::cout << move << std::endl;
    return move.empty() ? 1 : 0;
}
