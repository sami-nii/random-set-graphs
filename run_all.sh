

python main.py --dataset chameleon --model vanilla --count 100    ;
python main.py --dataset patents --model vanilla --count 100    ;
python main.py --dataset arxiv --model vanilla --count 100    ;
python main.py --dataset reddit2 --model vanilla --count 100    ;
python main.py --dataset coauthor --model vanilla --count 100    ;
python main.py --dataset squirrel --model vanilla --count 100    ;

python main.py --dataset chameleon --model credal --count 30    ;
python main.py --dataset patents --model credal --count 30   ;
python main.py --dataset arxiv --model credal --count 30    ;
python main.py --dataset reddit2 --model credal --count 30    ;
python main.py --dataset coauthor --model credal --count 30    ;
python main.py --dataset squirrel --model credal --count 30    ;

python main.py --dataset chameleon --model credal_LJ --count 30    ;
python main.py --dataset patents --model credal_LJ --count 30    ;
python main.py --dataset arxiv --model credal_LJ --count 30    ;
python main.py --dataset reddit2 --model credal_LJ --count 30    ;
python main.py --dataset coauthor --model credal_LJ --count 30    ;
python main.py --dataset squirrel --model credal_LJ --count 30    ;

python main.py --dataset squirrel --model ensemble  ;
python main.py --dataset arxiv --model ensemble  ;  
python main.py --dataset coauthor --model ensemble  ;
python main.py --dataset reddit2 --model ensemble  ;
python main.py --dataset patents --model ensemble  ;
python main.py --dataset chameleon --model ensemble  ;
