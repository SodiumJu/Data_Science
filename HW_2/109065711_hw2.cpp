#include <fstream>  
#include <vector>    
#include <iostream>
#include <iomanip>      // std::setprecision
#include <unordered_map> // for hash
#include <map>
#include <set>
#include <queue>

using namespace std;

struct items_vec_f
{
	set<int, std::less<int> > items;
	int frequency;
	items_vec_f(){
		frequency=0;
	}
	items_vec_f(int key_, int frequency_)
    {
        items.insert(key_);
     	frequency=frequency_;
    }
    bool operator<(const items_vec_f& vec1) const
    {
    	set<int, std::less<int> >:: iterator itr1;
    	set<int, std::less<int> >:: iterator itr2;
    	if(this->items.size()==vec1.items.size()){
    		
	    	itr1 = vec1.items.begin();
	    	itr2 = this->items.begin();
	    	while(itr1 != vec1.items.end()){
	    		if(*itr1>*itr2){
	    			return true;
	    		}
	    		else if(*itr1<*itr2){
	    			return false;
	    		}
	    		else{
	    			itr1++;
	    			itr2++;
	    		}
	    	}
	    	return false;
    	}
    	else if(this->items.size()<vec1.items.size()){
    		return true;	 
    	}else{
    		return false;
    	}
      //return (this->id < t.id);
    }
};

typedef set<int, std::less<int> >int_set_t;
typedef set<int, std::less<int> >::size_type int_size_type;
typedef set<int_set_t> int_power_set_t;

//static int_set_t testSet = {1, 2, 3, 4, 5};


void find_comb(int_set_t &inSet,int temp_f,map<set<int, std::less<int> >, int> &comb_f)
{
    int_power_set_t retSet;
    int_set_t workSet;
    int_set_t::const_iterator setIt;
    int_set_t::const_iterator curIt;
    const uint32_t maxSubSize = inSet.size() - 1;
    uint32_t curSubSize = 0;
    uint32_t sizeCnt = 0;
    uint32_t curIdx = 0;
    uint32_t setCnt = 0;
    int32_t numSets = 0;

    //Power set consist of
    //Empty set
    //retSet.insert(workSet);
    if(!inSet.empty()){
    	//The set itself
	    retSet.insert(inSet);
	    comb_f[inSet]=comb_f[inSet]+temp_f;
	    //Each single item
	    //Each combination of items (NOT permutations)
	    workSet.clear();
	    curIdx = 0;
	    //For each element
	    for(setIt = inSet.begin(); setIt != inSet.end(); setIt++ ) {
	        //For each subset size
	        for(curSubSize = 1; curSubSize <= maxSubSize; curSubSize++) {
	            if(curSubSize == 1) { //Special case, just add single item
	                numSets = 1;
	            } else {
	                numSets = inSet.size() - curSubSize + 1 - curIdx;
	                //In the negative case, create 0 sets
	                numSets = (numSets < 0) ? 0 : numSets;
	            }
	            //Number of sets of this size
	            for(setCnt = 0; setCnt < numSets; setCnt++) {
	                //Every set is based on the current item
	                workSet.insert(*setIt);
	                curIt = setIt;
	                advance(curIt, setCnt+1); //Advance to correct offset
	                //Number of elements in this size set
	                for(sizeCnt = 1; sizeCnt < curSubSize; sizeCnt++) {
	                    workSet.insert(*curIt);
	                    curIt++;
	                }
	                retSet.insert(workSet);
	                comb_f[workSet]=comb_f[workSet]+temp_f;
	                workSet.clear();
	            }
	        }
	        curIdx++;
	    }
	}   
}


class Node
{
	public:
		int key;
		int frequency;
    	Node* parent;
    	Node* right;
    	vector<Node*>child;
    	Node(){
    		key=-1;
    		frequency=0;
    		parent=NULL;
    		right=NULL;
    	}
    	Node(int key_,int frequency_,Node* parent_){
    		key=key_;
    		frequency=frequency_;
    		parent=parent_;
    		right=NULL;
    	}	
};

Node *newNode(int key,int frequency,Node* parent)
{
    Node *temp = new Node(key,frequency,parent);
    return temp;
}

 // Utility function to create a new tree node

class header_table {
	public:
		int frequency;
		Node* first_Node;
		Node* last_Node;
		header_table(){
			frequency=0;
			first_Node=NULL;
			last_Node=NULL;
		}
};

// Global variables
float min_sup;
int tx_num;
float min_fre;
string input_file_name;
string output_file_name;
unordered_map<int, header_table> umap;
Node root; //tree root
vector<int> items;
set<items_vec_f> ans_vec;
// Global variables

void print_node(){
	int n;
	queue<Node*> q;
	q.push(&root);
	//Node* target=NULL;
	while (!q.empty()){
		n = q.size();
		while(n > 0){
			Node* p = q.front();
            q.pop();

            for (int i=0; i<p->child.size(); i++){
                q.push(p->child[i]);
                cout << p->child[i]->key << ":" << p->child[i]->frequency << " ";
            }
            if(p->child.size()>0){
            	cout << endl;
            }else{
            	cout << "*" << endl;
            }
            
            n--;
		}
	}
}

int find_e_node(Node* pre_node,int item){
	for(int i=0; i<pre_node->child.size(); i++){
		if(pre_node->child[i]->key==item){
			return i;
		}
	}
	return -1;
}

Node* insert_node(Node* pre_node,int item){
	Node* re_node;
	//Node* new_node;
	//cout << "pre" << endl;
	int index = find_e_node(pre_node,item);
	//cout << pre_node->key << " find: " << index << " " <<endl ;
	if(index!=(-1)){
		pre_node->child[index]->frequency++;
		return pre_node->child[index];
	}else{
		re_node=newNode(item,1,pre_node); //new node point to parent
		pre_node->child.push_back(re_node); //parent add 1 child
		if(umap[item].first_Node==NULL){
			umap[item].first_Node=re_node;
		}else{
			umap[item].last_Node->right=re_node;
		} //add to header table
		umap[item].last_Node=re_node;
		return re_node;
	}
}

void insert_one_tx(vector<int>& tx_list){
	Node* pre_node=&root;
	for(int i=0; i<tx_list.size(); i++){
		pre_node=insert_node(pre_node,tx_list[i]);
	}
}

bool compare_item(int item1, int item2) 
{ 
    if(umap[item1].frequency >= umap[item2].frequency){
      return 1;
    }else{
      return 0;
    }
}

void print_item_f(vector<int> tx_list){
	for(int j=0; j<tx_list.size(); j++){
      cout << tx_list[j] << ":" << umap[tx_list[j]].frequency << " ";
    }
    cout << endl;
} 

void read_file(string file_name,unordered_map<int, header_table>& umap){
	ifstream infile(file_name);
	string line;
	int item;
	int start =0;
	while(getline(infile, line)) {
		start=0;
		tx_num++;
		for(int i=0;i<line.length()+1;i++){	
			if(line[i]==',' || i==line.length()){
                    item=stoi(line.substr(start,i-start)); //substring to int
                    //reverse_index.push_back(item);
                    umap[item].frequency = umap[item].frequency+1;
                    //cout << item << " ";
                    start=i+1;
                  }
		}
	}
}

void bulid_tree(string file_name,unordered_map<int, header_table>& umap){
	ifstream infile(file_name);
	string line;
	int item;
	int start =0;
	vector<int> tx_list;
	while(getline(infile, line)) {

		start=0;
		for(int i=0;i<line.length()+1;i++){	
			if(line[i]==',' || i==line.length()){
                item=stoi(line.substr(start,i-start)); //substring to int
                if(umap[item].frequency>=min_fre){
                	tx_list.push_back(item);
                }
                start=i+1;
            }
		}

		//for one transaction
		sort(tx_list.begin(), tx_list.end(), compare_item);
		/*for(int i=0;i<tx_list.size();i++){
			cout << tx_list[i] << " ";
		}*/

		if(tx_list.size()!=0){
			/*cout << endl;*/
			insert_one_tx(tx_list);
			//print_item_f(tx_list);
			tx_list.clear();
		}
		

		//for one transaction
	}
}

void tracing_node_right(Node* first_Node,vector<Node*>& bottom_nodes){
	Node* tmp_Node_a;
	tmp_Node_a=first_Node;
	bottom_nodes.push_back(tmp_Node_a);
	while(tmp_Node_a->right!=NULL){
		bottom_nodes.push_back(tmp_Node_a->right);
		tmp_Node_a=tmp_Node_a->right;
	}
}

void mining_one_item(int item){
	map<set<int, std::less<int> >, int> comb_f;
	vector<Node*> bottom_nodes;
	items_vec_f subtree;
	vector<items_vec_f> Subtrees;
	vector<int> comb;
	Node* tmp_Node;
	int item_c;

	unsigned int pow_set_size;
	int counter, j,i;
	int temp_f;

	tracing_node_right(umap[item].first_Node, bottom_nodes);
	for(i=0; i<bottom_nodes.size(); i++){
		tmp_Node=bottom_nodes[i]->parent;
		item_c=bottom_nodes[i]->frequency;
		while(tmp_Node->parent!=NULL){
			subtree.frequency=item_c;
			subtree.items.insert(tmp_Node->key);
			tmp_Node=tmp_Node->parent;
		}
		Subtrees.push_back(subtree);
		subtree.items.clear();
	}
	for(i=0; i<Subtrees.size(); i++){
		temp_f=Subtrees[i].frequency;
		find_comb(Subtrees[i].items,temp_f,comb_f);
	}

	map<set<int, std::less<int> >, int>:: iterator itr;

	for(itr = comb_f.begin() ; itr != comb_f.end() ; ){
	    if(itr->second<min_fre){
	      itr = comb_f.erase(itr);
	    }
	    else{	      
			subtree.items=itr->first;			
			subtree.items.insert(item);
			subtree.frequency=itr->second;  
			ans_vec.insert(subtree);
	    	subtree.items.clear();
	    	itr++;
	    }	    
	}
	//single item
	ans_vec.insert(items_vec_f(item,umap[item].frequency));
}

void mining_tree(){
	unordered_map<int, header_table>:: iterator itr;

	/*for (itr = umap.begin(); itr != umap.end(); itr++)
    {	
    	if(itr->second.frequency<min_fre){
    		//cout << "Delete " << itr->first << endl;
    		umap.erase(itr->first);
    	}   	
    }*/

    //while (itr != umap.end()) {
    for(itr = umap.begin() ; itr != umap.end() ; ){
	   if(itr->second.frequency<min_fre){

	   	  //cout << "Delete " << itr->first << " " << itr->second.frequency << endl;
	      itr = umap.erase(itr);
	   }
	   else{	      
	      //cout << itr->first << "  " << itr->second.frequency << endl;
	      items.push_back(itr->first); 
	      itr++;
	   }
	}

    //cout << "In umap size: " << umap.size() << endl;
    sort(items.begin(), items.end(), compare_item);

    for(int i=items.size()-1; i>=0; i--){
    	//cout << items[i] << " ";
    	//mining tree for one item
    	mining_one_item(items[i]);
    } //Error
    cout << endl;
    //cout << items.size() << endl;
}

int main(int argc, char* argv[]) 
{ 	
	tx_num=0;
	min_sup=0.1; //default
	input_file_name="input1.txt";
	output_file_name="ouput1.txt";
	if(argc!=4){
		cout << "It needs 3 parameters." << endl;
	}else{
		min_sup=stof(argv[1]);
	    input_file_name = argv[2];
	    output_file_name = argv[3];

	    /*cout << min_sup << endl;

	    cout << input_file_name << endl;
	    cout << output_file_name << endl;
*/
		read_file(input_file_name,umap);
		min_fre=tx_num*min_sup;
		bulid_tree(input_file_name,umap);
		//print_node();
	    mining_tree();

		set<items_vec_f>:: iterator itr;
		set<int>:: iterator itr2;
		float temp_min;

		ofstream outfile;
		outfile.open(output_file_name, ios::out);

	    for (itr = ans_vec.begin(); itr != ans_vec.end(); itr++)
	    {	
	    	itr2 = itr->items.begin();
	    	if(itr2 != itr->items.end()){
	    		outfile << *itr2;
	    		//fprintf (outfile,"%d", *itr2);
	    		itr2++;
	    		while(itr2 != itr->items.end()){
		    		outfile << "," << *itr2;
		    		//fprintf (outfile,",%d", *itr2);
		    		//std::cout << std::fixed;
		    		itr2++;
		    	}
		    	temp_min=((float)itr->frequency)/tx_num;

		    	//outfile << std::fixed;

		    	outfile << ":" << fixed << setprecision(4) << temp_min <<endl; 
		    	//fprintf(outfile,":%.4f\n", temp_min);
	    	}   		
	    }    
	    //fclose(outfile);
	    outfile.close();
	}    
	return 0;
}
