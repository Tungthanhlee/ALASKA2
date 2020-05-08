USER2="dgx2"
USER1="dgx1"
QUAD="medical"
# push code to server
#dgx2
# rsync -vr $USER2:/data/tung/alaska/weights .

# #dgx1
# rsync -vr $USER1:/data/tung/alaska/weights .

#medical
rsync -vr $QUAD:/home/dev/tung/alaska/weights .