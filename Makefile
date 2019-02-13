make init:
	pip3 install -r requirements.txt
	chmod +x ./init.sh
	./init.sh

check_clean:
	@echo "Are you sure (this will delete all data files) ? [y/N]" && read ans && [ $${ans:-N} == y ]

clean:	check_clean
	rm -rf data/
