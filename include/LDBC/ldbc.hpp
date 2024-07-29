#pragma once
#include <CPU_adj_list/CPU_adj_list.hpp>

template <typename weight_type>
class LDBC : public graph_structure<weight_type> {
    public:

    LDBC() : graph_structure<weight_type>() {}
    LDBC(int n) : graph_structure<weight_type>(n) {}

	bool sup_bfs = false;
	bool sup_cdlp = false;
	bool sup_pr = false;
	bool sup_wcc = false;
	bool sup_sssp = false;
	std::string bfs_src_name;//get bfs vertex source
	std::string sssp_src_name;//get sssp vertex source
	std::string vertex_file, edge_file;
	int bfs_src = 0;//define bfs vertex source is 0
	int cdlp_max_its = 10;//cdlp algo max iterator num
	int pr_its = 10;//pr algo iterator num
	int sssp_src = 0;//define sssp vertex source is 0
	double pr_damping = 0.85;//pr algorithm damping coefficient

	void load_graph();
	void read_config(std::string config_path);

	void save_to_CSV(std::vector<std::pair<std::string, std::string>>& res, std::string file_path, std::string env_type);
};

template <typename weight_type>
void LDBC<weight_type>::read_config(std::string config_path) {
	std::ifstream file(config_path);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << config_path << std::endl;
        return;
    }

	std::cout << "Reading config file..." << std::endl;

    while (getline(file, line)) {
		if (line.empty() || line[0] == '#')
			continue;

		auto split_str = parse_string(line, " = ");

		if (split_str.size() != 2) {
			std::cerr << "Invalid line: " << line << std::endl;
			continue;
		}

        auto key = split_str[0];
		auto value = split_str[1];

        auto parts = parse_string(key, ".");
        if (parts.size() >= 2) {
			if (parts.back() == "vertex-file") {//Reading *.properties file to get vertex file. eg. datagen-7_5-fb.v
				vertex_file = value;
				std::cout << "vertex_file: " << vertex_file << std::endl;
			}
			else if (parts.back() == "edge-file") {
				edge_file = value;
				std::cout << "edge_file: " << edge_file << std::endl;
			}
			else if (parts.back() == "vertices") {
				this->V = stoi(value);
				std::cout << "V: " << this->V << std::endl;
			}
			else if (parts.back() == "edges") {
				this->E = stoi(value);
				std::cout << "E: " << this->E << std::endl;
			}
			else if (parts.back() == "directed") {
				if (value == "false")
					this->is_directed = false;
				else
					this->is_directed = true;
				std::cout << "is_directed: " << this->is_directed << std::endl;
			}
			else if (parts.back() == "names") {//eg. graph.datagen-7_5-fb.edge-properties.names = weight
				if (value == "weight")
					this->is_weight = true;
				else
					this->is_weight = false;
				std::cout << "is_weight: " << this->is_weight << std::endl;
			}//Gets the type of algorithm contained in the configuration file
			else if (parts.back() == "algorithms") {
				auto algorithms = parse_string(value, ", ");
				for (auto& algorithm : algorithms) {
					if (algorithm == "bfs")
						sup_bfs = true;
					else if (algorithm == "cdlp")
						sup_cdlp = true;
					else if (algorithm == "pr")
						sup_pr = true;
					else if (algorithm == "sssp")
						sup_sssp = true;
					else if (algorithm == "wcc")
						sup_wcc = true;
				}
				std::cout << "bfs: " << sup_bfs << std::endl;
				std::cout << "cdlp: " << sup_cdlp << std::endl;
				std::cout << "pr: " << sup_pr << std::endl;
				std::cout << "sssp: " << sup_sssp << std::endl;
				std::cout << "wcc: " << sup_wcc << std::endl;
			}
			else if (parts.back() == "cdlp-max-iterations") {
				cdlp_max_its = stoi(value);
				std::cout << "cdlp_max_its: " << cdlp_max_its << std::endl;
			}
			else if (parts.back() == "pr-damping-factor") {
				pr_damping = stod(value);
				std::cout << "pr_damping: " << pr_damping << std::endl;
			}
			else if (parts.back() == "pr-num-iterations") {
				pr_its = stoi(value);
				std::cout << "pr_its: " << pr_its << std::endl;
			}
			else if (parts.back() == "sssp-weight-property") {
				if (value == "weight")
					this->is_sssp_weight = true;
				else
					this->is_sssp_weight = false;
				std::cout << "is_sssp_weight: " << this->is_sssp_weight << std::endl;
			}
			else if (parts.back() == "max-iterations") {
				cdlp_max_its = stoi(value);
				std::cout << "cdlp_max_its: " << cdlp_max_its << std::endl;
			}
			else if (parts.back() == "damping-factor") {
				pr_damping = stod(value);
				std::cout << "pr_damping: " << pr_damping << std::endl;
			}
			else if (parts.back() == "num-iterations") {
				pr_its = stoi(value);
				std::cout << "pr_its: " << pr_its << std::endl;
			}
			else if (parts.back() == "weight-property") {
				if (value == "weight")
					this->is_sssp_weight = true;
				else
					this->is_sssp_weight = false;
				std::cout << "is_sssp_weight: " << this->is_sssp_weight << std::endl;
			}
            else if (parts.back() == "source-vertex") {
				if (parts[parts.size() - 2] == "bfs") {
					bfs_src_name = value;//get bfs source vertex; eg. graph.datagen-7_5-fb.bfs.source-vertex = 6
					std::cout << "bfs_source_vertex: " << value << std::endl;
				}
				else {
					sssp_src_name = value;//get sssp source vertex; eg. graph.datagen-7_5-fb.sssp.source-vertex = 6
					std::cout << "sssp_source_vertex: " << value  << std::endl;
				}
            }
        }
    }
	std::cout << "Done." << std::endl; 
    file.close();
}

template <typename weight_type>
void LDBC<weight_type>::load_graph() {
	this->clear();

	std::string vertex_file_path;
	std::cout << "Please input the vertex file path: ";
	std::cin >> vertex_file_path;

	std::cout << "Loading vertices..." << std::endl;
	std::string line_content;
	std::ifstream myfile(vertex_file_path);

	if (myfile.is_open()) {
		while (getline(myfile, line_content))//read data line by line
			this->add_vertice(line_content);//Parsed the read data
		myfile.close();
	}
	else {
		std::cout << "Unable to open file " << vertex_file << std::endl
			<< "Please check the file location or file name." << std::endl;
		getchar();
		exit(1);
	}

	std::cout << "Done." << std::endl;
	if (sup_bfs) {
		if (this->vertex_str_to_id.find(bfs_src_name) == this->vertex_str_to_id.end()) {//bfs_src_name from read_configure
			std::cout << "Invalid source vertex for BFS" << std::endl;
			getchar();
			exit(1);
		}
		else
			bfs_src = this->vertex_str_to_id[bfs_src_name];
	}
		
	if (sup_sssp) {
		if (this->vertex_str_to_id.find(sssp_src_name) == this->vertex_str_to_id.end()) {//sssp_src_name from read_configure
			std::cout << "Invalid source vertex for SSSP" << std::endl;
			getchar();
			exit(1);
		}
		else
			sssp_src = this->vertex_str_to_id[sssp_src_name];
	}

	this->OUTs.resize(this->V);
	this->INs.resize(this->V);

	std::string edge_file_path;
	std::cout << "Please input the edge file path: ";
	std::cin >> edge_file_path;

	std::cout << "Loading edges..." << std::endl;
	myfile.open(edge_file_path);

	if (myfile.is_open()) {
		while (getline(myfile, line_content)) {
			std::vector<std::string> Parsed_content = parse_string(line_content, " ");
			int v1 = this->add_vertice(Parsed_content[0]);//get 1st vertex
			int v2 = this->add_vertice(Parsed_content[1]);//get 2nd vertex
			weight_type ec = Parsed_content.size() > 2 ? std::stod(Parsed_content[2]) : 1;//get weight
			//graph_structure<weight_type>::add_edge(v1, v2, ec);
			this->fast_add_edge(v1, v2, ec);
		}
		myfile.close();
	}
	else {
		std::cout << "Unable to open file " << edge_file << std::endl
			<< "Please check the file location or file name." << std::endl;
		getchar();
		exit(1);
	}
	std::cout << "Done." << std::endl;
}

template <typename weight_type>
void LDBC<weight_type>::save_to_CSV(std::vector<std::pair<std::string, std::string>>& res, std::string file_path, std::string env_type) {
	std::ofstream out(file_path);

	std::string data_name = this->vertex_file;

	out << data_name << "," << env_type << std::endl;

	for (auto i : res)
		out << i.first << "," << i.second << std::endl;

	out.close();
}
