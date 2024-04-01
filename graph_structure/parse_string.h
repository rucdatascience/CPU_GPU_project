#pragma once
#include<vector>
#include<string>

std::vector<std::string> parse_string(std::string parse_target, std::string delimiter) {

	std::vector<std::string> Parsed_content;
	size_t pos = 0;
	std::string token;
	while ((pos = parse_target.find(delimiter)) != std::string::npos) {
		// find(const string& str, size_t pos = 0) function returns the position of the first occurrence of str in the string, or npos if the string is not found.
		token = parse_target.substr(0, pos);
		// The substr(size_t pos = 0, size_t n = npos) function returns a substring of the object, starting at position pos and of length npos
		Parsed_content.push_back(token); // store the subtr to the list
		parse_target.erase(0, pos + delimiter.length()); // remove the front substr and the first delimiter
	}
	Parsed_content.push_back(parse_target); // store the subtr to the list

	return Parsed_content;

}


/*
-----------
#include <text_mining/parse_string.h>

int main()
{
	example_parse_string();
}
----------------
*/

#include<iostream>
void example_parse_string() {

	std::string s = "sfgdssddd";
	auto xx = parse_string(s, "gd");
	std::cout << xx[0] << "|" << xx[1] << std::endl;
}