/***********************************************
SFM code: fw.c   (version 1.1) (Jan 2011)
written by Shigueo Isotani and Satoru Fujishige
***********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "quicksort.c"

#define TRUE  1
#define FALSE 0

#define EPS 1.0E-10  /** allowed computational error bound **/
#define resolution 1.0  
 /** lower bound for nonzero values of |f(X)-f(Y)| for Y \subset X;
     default value 1.0 for integer valued submodular function f **/ 

int number_extreme_point = 0; /** the number of generated extreme bases **/

/* memory allocation  */
static void *xmalloc( size_t size );
int* ivectoralloc(int n);
double* vectoralloc(int n);
int** imatrixalloc(int m, int n);
double** matrixalloc(int m, int n);

int
func_fw(int n, double sf(int *p, int j, int n), double *x)
{
  int i, ih, it, ic, ix, ixh, ii, j, jj, kx;
  int jk, k, kk, ipk, flag, bool, ik;
  double sf_new, sf_previous;
  double theta, pp, xx, xpj, rr, usum, aid, bid, cid, rid, xxp, ss;
    
  int *ip;
  int *ib;
  int *s;
  double *xz;
  double *u; 
  double *ubar;
  double *v;
  double *w;
  double *pss;  
  double **r;
  double **ps;
  
  ip  = ivectoralloc(n+2);
  ib  = ivectoralloc(n+2);
  s   = ivectoralloc(n+1);
  xz  = vectoralloc(n+1);
  u   = vectoralloc(n+1);
  ubar= vectoralloc(n+1);
  v   = vectoralloc(n+1);
  w   = vectoralloc(n+1);
  pss = vectoralloc(n+2);
  r   = matrixalloc(n+2,n+2);
  ps  = matrixalloc(n+2,n+2);  

  for (j = 1; j <= n + 1; j++){
    ip[j] = j+1;
    ib[j] = 0;
  }
  ip[n+1] = 0;
  
  kx = 0;
  xxp = 1.0E+10;
  ixh = 0;
  
  for (j = 1; j <= n; j++){
    x[j]  = 0.0;
  }

  for(j = 0; j <= n; j++){
      s[j] = j;
  }
  /**************   step 0 *********************/
  xx = 1.0;
  sf_previous = 0.0;
  for (j = 1; j <= n ; j++){
    sf_new = sf(s,j,n);
    ps[1][j] = sf_new - sf_previous;
    sf_previous = sf_new;
    xx += ps[1][j]*ps[1][j];
  }
  pss[1] = xx - 1.0;
  r[1][1] = sqrt(xx);
  ih = it = w[1] = 1;
  ic = 2;
  ip[ih] = 0;
  k = 1;
  
  /**************   step 1 *********************/
  
  /******************* (A) ********************/
  
  while(1){/* 1000 */
    for (j = 1; j <= n; j++){
      xz[j]  = x[j];
      x[j] = 0.0;
    }
    
    i = ih;
    while( i != 0){
      for(j = 1; j <= n; j++){
	x[j]  += ps[i][j]*w[i];
      }
      i = ip[i];
    }

    /******************* (B) ********************/
    
    /* sort x[1..n] using quicksort algorithm:  
       s[i] contains the index of i-th smallest value of x */
    for(j = 0; j <= n; j++){
      s[j] = j;
    }   
    quicksort2(x,s,1,n);
    
    /* generate a new extreme base */
    sf_previous = 0.0;
    for (j = 1; j <= n ; j++){
      sf_new = sf(s,j,n);      
      ps[ic][s[j]] =  sf_new - sf_previous;
      sf_previous = sf_new;
    }
    
    number_extreme_point++;

    /******************* (C) ********************/
    xx = xpj = pp = 0.0;
    
    for (j = 1; j <= n; j++){
      pp += ps[ic][j] * ps[ic][j];
      xx += x[j]*x[j];
      xpj += x[j] * ps[ic][j];
    }
    pss[ic] = pp;
    
    if ( (xxp - xpj) <= EPS){
      kx++;
      if (kx >= 10) return 0;
    }


    i = ih;
    ss = 1.0;
    while( i != 0){
      if (ss < pss[i]){ ss = pss[i];}
      i = ip[i];
    }
    if (ss < pp) ss = pp;


    xxp = xx;
    if ((xx - xpj) < (ss * EPS) ){ 
      //printf("exit xx = xpj \nSize of the final corral = %d\n", k);
      return 0; 
    }

    if (k == n){
      printf("Trying to augment full-dimensional S!\n ");
      return (0);
    }
    
    /******************* (D) ********************/
  
    i = ih;
    while ( i != 0){
      flag = 0;
      for (j =  1; j <= n; j++){
	if( ((ps[i][j] - ps[ic][j])*(ps[i][j] - ps[ic][j])) > EPS ) {
	  i = ip[i];
	  flag = 1;
	  break;
	}
      }      
      if (flag == 0) break;
    }
      if (!flag) { printf("The generated point is in S!\n"); return 0; }
    
    /******************* (E), (F), (G) ********************/
    rr = 0.0;
    i  = ih;
    for ( jk = 1; jk <= k; jk++){
      r[ic][jk] = 1.0;
      for (j = 1; j  <= n; j++){
	r[ic][jk] += ps[i][j]*ps[ic][j];
      }
      for (jj = 1; jj <= jk -1; jj++){
	r[ic][jk] -=  r[i][jj]*r[ic][jj];
      }
      r[ic][jk]  = r[ic][jk]/r[i][jk];
      rr += r[ic][jk]*r[ic][jk];
      i = ip[i];
    }

    r[ic][k+1] = sqrt(1+pp-rr);
    
    k++;
    ip[it] = ic;
    ib[ic] = it;
    it = ic;
    ic = ip[ic];
    ip[it] = 0;

    w[it] = 0.0;

    
    
    /**************   step 2 *********************/
    while(1){ /* 2000 */
      i = ih;
      for (j = 1; j  <= k; j++){
	ubar[j] = 1.0;
	for (jj = 1;jj <= j-1; jj++){
	  ubar[j]  -=  ubar[jj]*r[i][jj];
	}
	ubar[j] = ubar[j]/r[i][j];
	i = ip[i];
      }
      
      usum = 0.0;
      i = it;
      for (j = k; j >= 1; j--){
	u[i] = ubar[j];
	ii = ip[i];
	while (ii != 0){
	  u[i] = u[i] - r[ii][j]*u[ii];
	  ii = ip[ii];
	}
	u[i] = u[i]/r[i][j];
	usum += u[i];
	i  = ib[i];
      }
      
      bool = TRUE;
      i = ih;
      while  (i != 0){
	v[i] = u[i]/usum;
	if (v[i] < EPS) bool = FALSE;
	i = ip[i];
      }
      
      if (bool) {
	i = ih;
	while (i != 0){
	  w[i] = v[i];
	  i = ip[i];
	}
	break;
      }
      /**************   step 3 *********************/
  
      /******************* (A), (B) ********************/
      theta = 1.0;
      i = ih;
      while (i != 0){
	if ( (w[i] -v[i]) > EPS)
	  theta = (theta <= w[i]/(w[i]- v[i]) )? theta : w[i]/(w[i] -v[i]) ;
	i = ip[i];
      }
      
      /******************* (C) ********************/
      kk = 0;
      i = ih;
      for (j=1; j <= k; j++){
	w[i] = (1-theta)*w[i] + theta*v[i];
	ipk = ip[i];
	
	/******************* (D), (E), (F), (G) ********************/
	
	if( w[i] < EPS ){
	  w[i] = 0.0;
	  kk++;
	  ii = ip[i];
	  for  (jj  = j -kk +1; jj <= k -kk; jj++){
	    aid = r[ii][jj];
	    bid  = r[ii][jj +1];
	    cid = sqrt (aid*aid + bid*bid);
	    ix = ii;
	    while (ix != 0){
	      rid = r[ix][jj];
	      r[ix][jj]  =  (aid * rid + bid * r[ix][jj+1])/ cid;
	      r[ix][jj+1]  =  (-bid * rid + aid * r[ix][jj+1])/ cid;
	      ix  = ip[ix];
	    }
	    ii = ip[ii];
	  }
	  
	  if ( i == ih){
	    ih = ip[i];
	    ib[ih] = 0;
	  }
	  else if (i == it){
	    it = ib[i];
	    ip[it] =  0;
	  }
	  else{
	    ip[ib[i]] = ip[i];
	    ib[ip[i]] = ib[i];
	  }
	  ip[i] = ic;
	  ib[i] = 0;
	  ic = i;
	}
	i = ipk;
      }
      k = k - kk;

    }  
    
  }     
  
  free(ip);
  free(ib);
  free(s) ;
  free(xz);
  free(u) ;
  free(ubar);
  free(v) ;
  free(w)  ;
  free(pss);
  free(r)  ;
  free(ps) ; 
  
  return 0;
}

static void *xmalloc( size_t size ) {
    void *s;

    if ( size == 0 ) size++;
    if ( (s=malloc(size)) == NULL ) {
        fprintf( stderr, "malloc : Not enough memory.\n" );
        exit( EXIT_FAILURE );
    }
    return s;
}

int* ivectoralloc(int n)   {  return xmalloc(sizeof(int) * n);  }

double* vectoralloc(int n) {  return xmalloc(sizeof(double) * n);  }

int** imatrixalloc(int m, int n) {
    int **a;
    int i, size2, size1;
    char *p;

    size2 = sizeof(*a)  * m;
    size1 = sizeof(**a) * n;
    p = xmalloc(size2 + size1 * m);

    a = (int**)p;  p += size2;
    for (i=0; i<m; i++) {
        a[i] = (int*)p;  p += size1;
    }
    return a;
}

double** matrixalloc(int m, int n) {
    double **a;
    int i, size2, size1;
    char *p;

    size2 = sizeof(*a)  * m;
    size1 = sizeof(**a) * n;
    p = xmalloc(size2 + size1 * m);

    a = (double**)p;  p += size2;
    for (i=0; i<m; i++) {
        a[i] = (double*)p;  p += size1;
    }
    return a;
}



int
mnp2sfm(int n, double sf(int *p, int j, int n), double *x)
{
  double gamma, fplus, fminus, eps0;
  double new, previous;
  int i,j,jj;
  int jmax,jmin;  /** the sizes of the max and the min minimizers **/
  int *smax, *smin; /** the indices of the max and the min minimizers **/
  int *s;

  s =  ivectoralloc(n+1);
  smax = ivectoralloc(n+1);
  smin = ivectoralloc(n+1);

  /* \sum { x(v) | x(v) < 0} */
  gamma=0.0;
  for (i = 1; i <= n; i++){
    if (x[i] < 0){
      gamma += x[i];
    }
    /**    printf("x[%d] = %8.15f\n",i, x[i]);**/ 
  }
    

  /* S^+ = {u | x(u) \leq 0} */
  for(i = 0; i <= n; i++){
    smax[i] = 0;
  }

  eps0 = ((double) resolution / n);
  /* Tolerance for computing minimizers from MNP x */    

  fplus = previous = 0.0;
  for (i = 1, j = 1 , jmax = 0; i <= n ; i++){
    if (x[i] <= eps0){
      smax[j] = i;
      jmax = j; j++;
    }
  }
  printf("|max_minimizer(S^+)| = %d\n",jmax);/***********/ 
  fplus = sf(smax, jmax, n);

  /* S^- = {u | x(u) < 0} */
  for(i = 0; i <= n; i++){
    smin[i] = 0;
  }
  fminus = previous = 0.0;
  for (i = 1, j = 1, jmin = 0 ; i <= n ; i++){
    if (x[i] < -eps0){
      smin[j] = i; 
      jmin = j; j++;
    }
  }
  printf("|min_minimizer(S^-)| = %d\n",jmin);/***********/ 
  fminus = sf(smin, jmin, n);  
  
  printf("++++++++++++++++++++++++++\n");
  printf("x^-(V) = %8.6f\n", gamma);
  printf("f(S^+) = %8.6f\n", fplus);
  printf("f(S^-) = %8.6f\n", fminus);
   
  printf("\nMaximum Minimizer\n");
  printf("S^+ = {");
  for (i=1; i<=jmax; i++){
    printf("%d", smax[i]);
    if (i!=jmax) printf(",");
  }
  printf("}\n");

  printf("\nMinimum Minimizer\n");
  printf("S^- = {");
  for (i=1; i<=jmin; i++){
    printf("%d", smin[i]);
    if (i!=jmin) printf(",");
  }
  printf("}\n");

  free(s);
  free(smax);
  free(smin);
  return 0;

}


double
func_minValue(int n, double sf(int *p, int j, int n), double *x)
{
  double gamma;
  int i;
 
  /* \sum { x(v) | x(v) < 0} */
  gamma=0.0;
  for (i = 1; i <= n; i++){
    if (x[i] < 0){
      gamma += x[i];
    }
  }
  return gamma;
}


int
func_smax(int n, double sf(int *p, int j, int n), double *x, int *smax)
{
  double eps0;
  double previous;
  int i,j;
  int jmax;  /** the size of the max minimizers **/
    
  /* S^+ = {u | x(u) \leq 0} */
  for(i = 0; i <= n; i++){
    smax[i] = 0;
  }

  eps0 = ((double) resolution / n);
  /* Tolerance for computing minimizers from MNP x */    

  previous = 0.0;
  for (i = 1, j = 1 , jmax = 0; i <= n ; i++){
    if (x[i] <= eps0){
      smax[j] = i;
      jmax = j; j++;
    }
  }
  
  return jmax; /* return the size of the max minimizer */
}


int
func_smin(int n, double sf(int *p, int j, int n), double *x, int *smin)
{
  double eps0;
  double previous;
  int i,j;
  int jmin;  /** the size of the min minimizers **/
      
  eps0 = ((double) resolution / n);
  /* Tolerance for computing minimizers from MNP x */    

  /* S^- = {u | x(u) < 0} */
  for(i = 0; i <= n; i++){
    smin[i] = 0;
  }
  previous = 0.0;
  for (i = 1, j = 1, jmin = 0 ; i <= n ; i++){
    if (x[i] < -eps0){
      smin[j] = i; 
      jmin = j; j++;
    }
  }
  
  return jmin; /* return the size of the min minimizer */
}
