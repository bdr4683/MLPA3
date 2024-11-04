all: 
	python a3_q1.py
	python a3_q2.py
	python a3_q3.py
	@echo "Finished."

q1:
	python a3_q1.py

q2:
	python a3_q2.py

q3:
	python a3_q3.py

install:
	pip3 install pandas scikit-learn 

clean:
	rm -f *.pdf

