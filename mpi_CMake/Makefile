PRG = HeatSolver
EXECUTABLE = $(PRG).exe
IDIR =./include
CC=mpicc
DEBUG = -g 
CFLAGS=-Wall -O2 -std=c99 $(DEBUG) -I$(IDIR) 

OBJECTDIR=./src/obj/

LDIR =./lib
SOURCES = ./src
LIBS=-lm 

_DEPS = myFunctions.h definitions.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJECTS = main.o myFunctions.o 
OBJECTS = $(patsubst %,$(OBJECTDIR)/%,$(_OBJECTS))


$(OBJECTDIR)/%.o: $(SOURCES)/%.c $(DEPS)
	@echo "Generating object files ..."
	@$(CC) -c -o $@ $< $(CFLAGS)

$(EXECUTABLE): $(OBJECTS)
	@echo "Generating the executable file " 
	@$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	@echo "Cleaning up .. " 
	@rm -f $(OBJECTDIR)/*.o *~ $(EXECUTABLE) *.txt


.PHONY:pdf

pdf:
	@echo " Generating PDFs files ... " 
	@a2ps Makefile $(SOURCES)/*.c $(DEPS) --pro=color \
	--columns=2 -E -o mpi2D.ps
	@ps2pdf mpi2D.ps 
	@rm -rf *.ps

.PHONY:tar

tar:
	@echo "Generating tar files .. "
	@tar -cvf Serial.code.tar.gz $(SOURCES) $(DEPS) Makefile *.pdf 

.PHONY:remove

remove:
	@echo " Removing all the PDFs, TAR.GZ files ... "
	@rm -rf *.tar.gz *.ps *.pdf

