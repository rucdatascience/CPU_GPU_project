#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono> // C++11时间库
#include <random>
#include <iomanip>
#include <map>

// 定义用户信息结构体
struct UserInfo {
    std::string id; // 将ID改为字符串类型
    std::string name;
};

// 定义用户表文件路径
 std::string user_file = "../data/users.txt";

// 函数：读取用户表文件内容到vector<UserInfo>中
std::vector<UserInfo> readUserFile() {
    std::vector<UserInfo> users;
    std::ifstream file(user_file);

    if (!file.is_open()) {
        // 如果文件不存在，则创建文件
        std::ofstream create_file(user_file);
        if (!create_file.is_open()) {
            std::cerr << "Error: Could not create file " << user_file << std::endl;
            return users; // 返回空vector
        }
        create_file.close();
        return users; // 返回空vector，因为文件刚刚创建，里面没有内容
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        UserInfo user;
        if (iss >> user.id >> user.name) {
            users.push_back(user);
        } else {
            std::cerr << "Error: Invalid format in file " << user_file << std::endl;
        }
    }

    file.close();
    return users;
}

// 函数：将用户信息写入用户表文件
void writeUserFile( std::vector<UserInfo>& users) {
    std::ofstream file(user_file);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << user_file << " for writing" << std::endl;
        return;
    }

    for ( auto& user : users) {
        file << user.id << " " << user.name << std::endl;
    }

    file.close();
}

// 函数：生成当前时间戳字符串
std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    return std::to_string(millis);
}

// 函数：生成随机的UUID
std::string generateUUID() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);

    const char* hex = "0123456789abcdef";
    std::stringstream ss;
    for (int i = 0; i < 36; ++i) {
        if (i == 8 || i == 13 || i == 18 || i == 23) {
            ss << '-';
        } else if (i == 14) {
            ss << '4';
        } else if (i == 19) {
            ss << hex[dis(gen) & 0x3 | 0x8];
        } else {
            ss << hex[dis(gen)];
        }
    }

    return ss.str();
}

// 函数：根据用户名查询用户ID，如果不存在则生成新的时间戳ID并写入文件
std::string getUserIdByName( std::vector<UserInfo>& users,  std::string& name) {
    auto it = std::find_if(users.begin(), users.end(),
                           [&name]( UserInfo& user) { return user.name == name; });

    if (it != users.end()) {
        return it->id; // 找到用户，返回其ID
    } else {
        // 生成新的时间戳ID
        std::string new_id = getCurrentTimestamp();
        // 使用UUID作为用户ID
        // std::string userId = generateUUID();
        UserInfo new_user = {new_id, name};
        users.push_back(new_user);

        // 更新文件
        writeUserFile(users);

        return new_id;
    }
}

std::string getUserNameById(std::string userId){
    std::ifstream file("../data/users.txt");
    if (!file) {
        std::cerr << "Error opening file." << std::endl;
    }

    std::map<std::string, std::string> userIdToName;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        UserInfo user;
        if (iss >> user.id >> user.name) {
            userIdToName[user.id] = user.name;
        } else {
            std::cerr << "Error parsing line: " << line << std::endl;
        }
    }

    file.close();

    auto it = userIdToName.find(userId);
    if (it != userIdToName.end()) {
        std::cout << "User ID " << userId << "<--------->" << it->second << std::endl;
        return it->second;
    } 
}
/**
int main() {
    std::string username;
    std::cout << "Enter username: ";
    std::cin >> username;

    // 读取用户表文件内容
    std::vector<UserInfo> users = readUserFile();

    // 查询用户ID
    std::string user_id = getUserIdByName(users, username);

    std::cout << "User ID: " << user_id << std::endl;

    std::string user_name = getUserNameById(user_id);

    std::cout << "User Name:" <<user_name <<std::endl;

    return 0;
}
*/