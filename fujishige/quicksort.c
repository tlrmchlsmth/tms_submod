void 
swap (A, i, j)
     double *A;
     int i;
     int j;
{
  double temp;
  
  temp = A[i];
  A[i]  = A[j];
  A[j] = temp;
}


int 
partition(A, left, right)
     double *A;
     int left;
     int right; 
{
  int i = left-1;    /* left to right pointer */
  int j = right;     /* right to left pointer */
  
  for(;;) {
    while (A[++i] < A[right]);   /*  find element on left to swap */
    while (A[right] < A[--j])    /* look for element on right to swap, but don't run off end */
      if (j == left)     
	break;

    if (i >= j)
      break;    /* pointers cross */
    swap(A, i, j);
  }
  swap(A, i, right);  /*  swap partition  element */ 
  return i;  
}

void
quicksort(A, left, right)
     double *A;
     int left;
     int right;
{
  int q;
  
  if (right  > left) {
    q = partition (A, left, right);
    quicksort(A, left, q-1);
    quicksort(A, q+1, right);
  }
}


/*******************************************************************************/

void 
swap2 (s, i, j)
     int *s;
     int i;
     int j;
{
  int temp;
  
  /* temp = A[i];
     A[i]  = A[j];
     A[j] = temp;*/
  
  temp = s[i];
  s[i] = s[j];
  s[j] = temp;
  
}

int 
partition2(A, s, left, right)
     double *A;
     int *s;
     int left;
     int right; 
{
  int i = left-1;    /* left to right pointer */
  int j = right;     /* right to left pointer */
  
  for(;;) {
    while (A[s[++i]] < A[s[right]]);   /*  find element on left to swap */
    while (A[s[right]] < A[s[--j]])    /* look for element on right to swap, but don't run off end */
      if (j == left)     
	break;
    
    if (i >= j)
      break;    /* pointers cross */
    swap2(s, i, j);
  } 
  swap2(s, i, right);  /*  swap partition  element */ 
  return i;  
}


void
quicksort2(A, s, left, right)
     double *A;
     int *s; /* permutation array */
     int left;
     int right;
{
  int q;
  
  if (right  > left) {
    q = partition2 (A, s, left, right);
    quicksort2(A, s, left, q-1);
    quicksort2(A, s,  q+1, right);
  }
}
