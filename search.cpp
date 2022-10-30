#include <bits/chrono.h>
#include <cmath>
#include <future>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <chrono>
#include <queue>
#include <stack>
#include "OleanderStemmingLibrary/include/olestem/stemming/english_stem.h"
#include "ThreadPool/ThreadPool.h"
#include "LRU.h"

using namespace std;
stemming::english_stem<> StemEnglish;
unordered_set<string> STOP_WORDS = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "you're", "you've", "you'll", "you'd", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "she's", "her", "hers", "herself", "it", "it's", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "that'll", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "don't", "should", "should've", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "aren't", "couldn", "couldn't", "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "isn", "isn't", "ma", "mightn", "mightn't", "mustn", "mustn't", "needn", "needn't", "shan", "shan't", "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't", "won", "won't", "wouldn", "wouldn't", "reflist", "refbegin", "ref", "citation", "refend", "references", "category", "infobox", "external", "links", "https", "http", "www", "org", "com", "edu", "jpg", "jpeg", "pdf", "png"};
unordered_set<char> FIELDS = {'t', 'b', 'c', 'i', 'l', 'r'};
long long QUERY_LENGTH = 0;
unordered_map<string, unsigned int> word_positions;
unordered_map<int, pair<unsigned int, int>> title_positions;
string word_positions_file, title_positions_file, INDEX_DIR;
string titles_file = "title_map";
int title_weight = 1000;
int category_weight = 500;
int infobox_weight = 300;
int rest_weight = 100;
int match_weight = 20000;
int K = 10;
ThreadPool pool(8);

struct DocData {
    int id;
    vector<pair<char, unsigned int>> data;
};

void getSecondaryIndices() {
    ifstream fin(word_positions_file);
    string line, word;
    unsigned int seek_value;
    int word_count, doc_id;
    unsigned long long start = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();
    while(getline(fin, line)) {
        istringstream inp(line);
        inp >> word >> seek_value;
        word_positions[word] = seek_value;
    }
    fin.close();
    fin.clear();
    fin.open(title_positions_file);

    while(getline(fin, line)) {
        istringstream inp(line);
        inp >> doc_id >> seek_value >> word_count;
        title_positions[doc_id] = {seek_value, word_count};
    }
    unsigned long long end = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();
    cout << "Done prefetching in " << (end - start) / (long double)1000 << '\n';
    cout << word_positions.size() << ' ' << title_positions.size() << '\n';
}



string get_title_data(int title_id) {
    if(title_positions.find(title_id) == title_positions.end()) {
        return "";
    }
    unsigned int seek_value = title_positions[title_id].first;
    ifstream fin(INDEX_DIR + "/" + titles_file);
    fin.seekg(seek_value);
    string title_data;
    getline(fin, title_data);
    int i;
    for(i = 0; i < title_data.size() && title_data[i] != ':'; i++) ;
    return title_data.substr(i + 1, title_data.size() - i - 1);
}

bool isUseful(string& token) {
    if(token.size() < 2 || STOP_WORDS.find(token) != STOP_WORDS.end())
        return false;
    if(token[0] >= '0' && token[0] <= '9')
        return false;
    if(token[1] >= '0' && token[1] <= '9')
        return false;
    int digitCount = 0, length = token.size();
    for(int i = 2; i < length; i++) {
        if(token[i] >= '0' && token[i] <= '9')
            ++digitCount;
    }
    return (2*digitCount >= length ? false : true);
}

void getQueries(vector<string>& queries, string query_file) {
    ifstream fin(query_file);
    string query;
    while(!fin.eof()) {
        getline(fin, query);
        if(query.size())
            queries.push_back(query += '\n');
    }
}

void query_processor(const string& query, vector<string>& tokens) {
    string token;
    for(const char& ch: query) {
        if(ch >= 0 && ch <= 127) {
            if(ch <= 'Z' && ch >= 'A')
                token += (ch - 'A') + 'a';
            else if((ch <= 'z' && ch >= 'a') || (ch <= '9' && ch >= '0'))
                token += ch;
            else {
                if(isUseful(token)) {
                    wstring word(token.begin(), token.end());
                    StemEnglish(word);
                    tokens.emplace_back(word.begin(), word.end());
                }
                token = "";
            }
        }
        else {
            if(isUseful(token)) {
                wstring word(token.begin(), token.end());
                StemEnglish(word);
                tokens.emplace_back(word.begin(), word.end());
            }
            token = "";
        }
    }
    if(isUseful(token)) {
        wstring word(token.begin(), token.end());
        StemEnglish(word);
        tokens.emplace_back(word.begin(), word.end());
    }
}

void split_query(const string& query, unordered_map<string, unordered_map<char, unsigned int>>& word_field_freq) {
    int idx = 0, start_idx = 0;
    QUERY_LENGTH = 0;
    char cur_field = 'n';
    string subquery;
    unordered_map<char, string> field_unordered_map;
    while(idx + 1 < query.size()) {
        if(FIELDS.find(query[idx]) != FIELDS.end() && query[idx + 1] == ':') {
            if(start_idx != idx) {
                if(field_unordered_map.find(cur_field) == field_unordered_map.end())
                    field_unordered_map[cur_field] = "";
                field_unordered_map[cur_field].append(subquery);
                subquery = "";
            }
            cur_field = query[idx++];
            start_idx = ++idx;
        } else
            subquery += query[idx++];
    }
    if(query.size())
        subquery += query.back();
    if(start_idx != idx) {
        if(field_unordered_map.find(cur_field) == field_unordered_map.end())
            field_unordered_map[cur_field] = "";
        field_unordered_map[cur_field].append(subquery);
    }
    for(auto& [field, subquery]: field_unordered_map) {
        vector<string> tokens;
        query_processor(subquery, tokens);
        for(const string& word: tokens) {
            if(word_field_freq.find(word) == word_field_freq.end())
                word_field_freq[word] = unordered_map<char, unsigned int>();
            if(word_field_freq[word].find(field) == word_field_freq[word].end())
                word_field_freq[word][field] = 0;
            ++word_field_freq[word][field];
            ++QUERY_LENGTH;
        }
    }
}

void split_into_docs(const string& posting_list, vector<DocData>& docs) {
    unsigned int idx = 0;
    string text;
    char field;
    DocData data;
    data.id = 0;
    while(idx < posting_list.size()) {
        if(posting_list[idx] == ';') {
            data.data.emplace_back(field, stoi(text));
            docs.push_back(data);
            text.clear();
            data.id = 0;
            data.data.clear();
        }
        else if(posting_list[idx] >= '0' && posting_list[idx] <= '9') {
            text += posting_list[idx];
        }
        else {
            if(!data.id)
                data.id = stoi(text);
            else
                data.data.emplace_back(field, stoi(text));
            field = posting_list[idx];
            text.clear();
        }
        ++idx;
    }
    data.data.emplace_back(field, stoi(text));
    docs.push_back(data);
}

void get_posting_list(const string& word, vector<DocData>& docs) {
    unordered_map<string, unsigned int>::iterator itr = word_positions.find(word);
    if(itr == word_positions.end())
        return ;
    unsigned int seek_value = itr->second;
    string file = INDEX_DIR + '/' + word.substr(0, 1);
    ifstream fin(file);
    string posting_list;
    fin.seekg(seek_value);
    getline(fin, posting_list);
    split_into_docs(posting_list, docs);
}

unordered_map<int, long double> get_tfidf_vector(const string& word, const unordered_map<char, unsigned int>& field_freq) {
    unordered_map<int, long double> tfidf_vector;
    vector<DocData> docs;
    get_posting_list(word, docs);
    if(!docs.size())
        return unordered_map<int, long double>();
    int word_in_docs_cnt = docs.size();
    long double idf = log10((title_positions.size() + (long double)0.00) / word_in_docs_cnt);
    for(const DocData& doc_data: docs) {
        if(!doc_data.data.size())
            return unordered_map<int, long double>();
        unsigned long long wc = 0, extra = 0, wc_in_doc = title_positions[doc_data.id].second;
        if(!wc_in_doc)
            return unordered_map<int, long double>();
        if(tfidf_vector.find(doc_data.id) == tfidf_vector.end())
            tfidf_vector[doc_data.id] = 0;
        for(const pair<char, unsigned int>& freq: doc_data.data) {
            if(freq.first == 't') {
                wc += freq.second * title_weight;
                extra += freq.second * title_weight;
            } else if(freq.first == 'c') {
                wc += freq.second * category_weight;
                extra += freq.second * category_weight;
            } else if(freq.first == 'i') {
                wc += freq.second * infobox_weight;
                extra += freq.second * infobox_weight;
            } else {
                wc += freq.second * rest_weight;
                extra += freq.second * rest_weight;
            }
            if(field_freq.find(freq.first) != field_freq.end()) {
                wc += freq.second * match_weight;
                extra += freq.second * match_weight;
            }
        }
        tfidf_vector[doc_data.id] += (wc / (wc_in_doc + extra + (long double)0.00)) * idf;
    }
    return tfidf_vector;
}

void get_tfidf_vectors(const unordered_map<string, unordered_map<char, unsigned int>>& word_field_freq, unordered_map<int, long double>& tfidf_vectors) {
    std::vector<std::future<unordered_map<int, long double>>> results;
    for(auto& [word, field_freq]: word_field_freq) {
        results.emplace_back(
            pool.enqueue(get_tfidf_vector, word, field_freq)
        );
    }
    for(auto && result: results) {
        unordered_map<int, long double> tfidf_vector = result.get();
        for(auto& [doc_id, score]: tfidf_vector) {
            unordered_map<int, long double>::iterator itr = tfidf_vectors.find(doc_id);
            if(itr == tfidf_vectors.end()) {
                tfidf_vectors[doc_id] = score;
            } else {
                itr->second += score;
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("ERR: Too Few Arguments\n");
        return 0;
    }
    title_positions_file = word_positions_file = INDEX_DIR = argv[1];
    title_positions_file.append("/title_position");
    word_positions_file.append("/word_position");
    getSecondaryIndices();
    while(true) {
        cout << "\n>";
        string query;
        getline(cin, query);
        unsigned long long start = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();
        priority_queue<pair<long double, int>> similarity;
        unordered_map<int, long double> tfidf_vectors;
        unordered_map<string, unordered_map<char, unsigned int>> word_field_freq;
        split_query(query, word_field_freq);
        get_tfidf_vectors(word_field_freq, tfidf_vectors);
        for(auto& [doc_id, score]: tfidf_vectors) {
            similarity.push({-1*score, doc_id});
            if(similarity.size() == (K + 1))
                similarity.pop();
        }
        unsigned long long end = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();
        stack<string> st;
        while(!similarity.empty()) {
            st.push(get_title_data(similarity.top().second));
            similarity.pop();
        }
        while(!st.empty()) {
            cout << st.top() << '\n';
            st.pop();
        }
        cout << "Time taken: " << (end - start) / (long double)1000 << "\n";

    }
    return 0;
}
