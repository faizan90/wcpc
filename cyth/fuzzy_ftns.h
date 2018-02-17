#pragma once
#include <math.h>
//#include <stdio.h>

typedef double DT_D;

DT_D inline get_tri_mu(const DT_D *g,
					   const DT_D *a,
					   const DT_D *b,
					   const DT_D *c) {

    DT_D mu_g;

    if (*g <= *a) {
        mu_g = 0.0;
        //printf("less than a");
    }
    
    else if ((*a < *g) && (*g <= *b)) {
        mu_g = (*g - *a) / (*b - *a);
        //printf("between a and b");
    }
    
    else if ((*b < *g) && (*g < *c)) {
        mu_g = (*g - *c) / (*b - *c);
        //printf("between b and c");
    }

    else if (*g >= *c) {
        mu_g = 0.0;
        //printf("greater than c");
    }

    else {
     	mu_g = NAN;
     	//printf("something else");
    }
 
    return mu_g;
}
