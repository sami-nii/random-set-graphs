

# python main.py --dataset chameleon --model vanilla --count 100    ;
# python main.py --dataset patents --model vanilla --count 100    ;
# python main.py --dataset arxiv --model vanilla --count 100    ;
# python main.py --dataset reddit2 --model vanilla --count 100    ;
# python main.py --dataset coauthor --model vanilla --count 100    ;
# python main.py --dataset squirrel --model vanilla --count 100    ;

# python main.py --dataset chameleon --model credal --count 30    ;
# python main.py --dataset patents --model credal --count 30   ;
# python main.py --dataset arxiv --model credal --count 30    ;
# python main.py --dataset reddit2 --model credal --count 30    ;
# python main.py --dataset coauthor --model credal --count 30    ;
# python main.py --dataset squirrel --model credal --count 30    ;

# python main.py --dataset chameleon --model credal_LJ --count 30    ;
# python main.py --dataset patents --model credal_LJ --count 30    ;
# python main.py --dataset arxiv --model credal_LJ --count 30    ;
# python main.py --dataset reddit2 --model credal_LJ --count 30    ;
# python main.py --dataset coauthor --model credal_LJ --count 30    ;
# python main.py --dataset squirrel --model credal_LJ --count 30    ;

# python main.py --dataset squirrel --model ensemble  ;
# python main.py --dataset arxiv --model ensemble  ;  
# python main.py --dataset coauthor --model ensemble  ;
# python main.py --dataset reddit2 --model ensemble  ;
# python main.py --dataset patents --model ensemble  ;
# python main.py --dataset chameleon --model ensemble  ;


# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset squirrel --model gebm  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset arxiv --model gebm  ;  
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset coauthor --model gebm  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset reddit2 --model gebm  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset patents --model gebm  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset chameleon --model gebm  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset amazon_ratings --model gebm  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset cora --model gebm  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset roman_empire --model gebm ;


# python main.py --dataset squirrel --model frozen -c 30;
# python main.py --dataset arxiv --model frozen -c 30;
# python main.py --dataset coauthor --model frozen -c 30;
# python main.py --dataset reddit2 --model frozen -c 30;
# python main.py --dataset patents --model frozen -c 30;
# python main.py --dataset chameleon --model frozen -c 30;
# python main.py --dataset amazon_ratings --model frozen_LJ -c 30;
# python main.py --dataset cora --model frozen_LJ -c 30;
# python main.py --dataset roman_empire --model frozen_LJ -c 30;


python main.py --dataset squirrel --model cagcn -c 30;
python main.py --dataset arxiv --model cagcn -c 30;
python main.py --dataset coauthor --model cagcn -c 30;
python main.py --dataset reddit2 --model cagcn -c 30;
python main.py --dataset patents --model cagcn -c 30;
python main.py --dataset chameleon --model cagcn -c 30;
python main.py --dataset amazon_ratings --model cagcn -c 30;
python main.py --dataset cora --model cagcn -c 30;
python main.py --dataset roman_empire --model cagcn -c 30;