int main(){
int x,A[101],B[101];
for(x=0;x<10;x++){
A[2*(x)]=2;
B[2*(x)+2]=A[x+4];
}
return 0;
}
