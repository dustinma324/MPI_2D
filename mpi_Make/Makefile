PROG = mpiHeat2D.exe

IDIR=./include
BINDIR=./bin
OBJDIR=./src/obj/
SOURCES=./src

TARGET = $(BINDIR)/$(PROG)

CC=mpicc
DEBUG = -g 
LIBS=-lm -lhdf5
CFLAGS=-Wall -O3 -std=c99 $(DEBUG) -I$(IDIR) 

_DEPS = myFunctions.h definitions.h outputHDF5.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJS = main.o myFunctions.o outputHDF5.o
OBJS = $(patsubst %,$(OBJDIR)/%,$(_OBJS))


$(OBJDIR)/%.o: $(SOURCES)/%.c $(DEPS)
	@echo "Generating object files ..."
	@$(CC) $(CFLAGS) -c -o $@ $<

$(TARGET): makedirectories $(OBJS)
	@echo "Generating the executable file " 
	@$(CC) $(CFLAGS) $(LIBS) -o $@ $(OBJS) 

.PHONY: clean

makedirectories:
	@mkdir -p $(OBJDIR) $(BINDIR)

clean:
	@echo "Cleaning up .. " 
	@rm -f $(OBJDIR)/*.o *~ $(TARGET) *.txt
	@rm -f $(BINDIR)/*.pdf


.PHONY:pdf

pdf:
	@echo " Generating PDFs files ... " 
	@a2ps Makefile $(SOURCES)/*.c $(DEPS) --pro=color \
	--columns=2 -E -o 2DMPI.ps
	@ps2pdf 2DMPI.ps 
	@rm -rf *.ps
	@mv *.pdf $(BINDIR)

.PHONY:tar

tar:
	@echo "Generating tar files .. "
	@tar -cvf Serial.code.tar.gz $(SOURCES) $(DEPS) Makefile *.pdf 

.PHONY:remove

remove:
	@echo " Removing all the PDFs, TAR.GZ files ... "
	@rm -rf *.tar.gz *.ps *.pdf
