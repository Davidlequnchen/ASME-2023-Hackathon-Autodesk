22700 0 2 4           
16 Autodesk Neutron 20 ASM 227.1.1.65535 NT 24 Thu Aug 26 00:43:35 2021 
10 9.999999999999999547e-07 1.000000000000000036e-10 
asmheader $-1 -1 @13 227.1.1.65535 #
body $2 6 $-1 $3 $-1 $4 #
ATTRIB_CUSTOM-attrib $-1 -1 $5 $-1 $1 @20 Timestamp_attrib_def 1 1455851548001 #
lump $-1 -1 $-1 $-1 $6 $1 #
transform $-1 -1 1 0 0 0 1 0 0 0 1 0 0 0 1 no_rotate no_reflect no_shear #
ATTRIB_CUSTOM-attrib $-1 -1 $-1 $2 $1 @22 generic_tag_attrib_def 3 3 -1 @23 generic_tag_attrib_def  1 3 @3 581 1 0 0 #
shell $-1 -1 $-1 $-1 $-1 $7 $-1 $3 #
face $8 0 $-1 $9 $10 $6 $-1 $11 forward single #
ATTRIB_CUSTOM-attrib $12 -1 $-1 $-1 $7 @22 generic_tag_attrib_def 3 3 -1 @23 generic_tag_attrib_def  1 1 @1 3 0 2 546 581 0 #
face $13 1 $-1 $14 $15 $6 $-1 $16 forward single #
loop $-1 -1 $-1 $-1 $17 $7 #
spline-surface $-1 -1 $-1 forward { helix_spl_circ 22140 F -3.141592653589793116 F 3.141592653589793116 
	F 0 F 18.8495559215387587 
	3.141592653589793116 
	F 0 F 18.8495559215387587 
	4.950000000000000178 1.650000000000000355 -0.500000000000000222 
	-0.4000000000000003553 0 0 
	0 0.4000000000000003553 0 
	0 0 -0.1166666666666666824 
	0 
	0 0 -1 
	null_surface 
	null_surface 
	nullbs 
	nullbs 
	
	0.04000000000000003553 
	} I I I I #
ATTRIB_CUSTOM-attrib $-1 -1 $-1 $-1 $8 @22 generic_tag_attrib_def 3 3 -1 @23 generic_tag_attrib_def  1 0 @0  0 1 547 0 #
ATTRIB_CUSTOM-attrib $18 -1 $-1 $-1 $9 @22 generic_tag_attrib_def 3 3 -1 @23 generic_tag_attrib_def  1 1 @1 2 0 2 546 581 0 #
face $19 2 $-1 $-1 $20 $6 $-1 $21 reversed single #
loop $-1 -1 $-1 $-1 $22 $9 #
plane-surface $-1 -1 $-1 4.549999999999998934 1.649999999999999911 -0.8500000000000003109 -7.34788079488411875e-16 1 0 0 0 1 forward_v I I I I #
coedge $-1 -1 $-1 $23 $24 $25 $26 forward $10 0 $27 #
ATTRIB_CUSTOM-attrib $-1 -1 $-1 $-1 $13 @22 generic_tag_attrib_def 3 3 -1 @23 generic_tag_attrib_def  1 0 @0  0 1 547 0 #
ATTRIB_CUSTOM-attrib $28 -1 $-1 $-1 $14 @22 generic_tag_attrib_def 3 3 -1 @23 generic_tag_attrib_def  1 1 @1 1 0 2 546 581 0 #
loop $-1 -1 $-1 $-1 $25 $14 #
plane-surface $-1 -1 $-1 4.549999999999998934 1.650000000000000133 -0.500000000000000222 0 1 0 0 0 1 forward_v I I I I #
coedge $-1 -1 $-1 $22 $22 $29 $30 forward $15 0 $-1 #
coedge $-1 -1 $-1 $29 $17 $24 $31 forward $10 0 $32 #
coedge $-1 -1 $-1 $17 $29 $23 $31 reversed $10 0 $33 #
coedge $-1 -1 $-1 $25 $25 $17 $26 reversed $20 0 $-1 #
edge $-1 3 $-1 $34 -3.141592653589793116 $34 3.141592653589793116 $17 $35 forward @7 unknown #
pcurve $-1 -1 $-1 0 forward { exp_par_cur nubs 1 periodic 2 
	-3.141592653589793116 1 3.141592653589793116 1 
	-3.141592653589793116 0 
	3.141592653589793116 0 
	0 
	spline forward { ref 0 } I I I I 
	} 0 0 #
ATTRIB_CUSTOM-attrib $-1 -1 $-1 $-1 $19 @22 generic_tag_attrib_def 3 3 -1 @23 generic_tag_attrib_def  1 0 @0  0 1 547 0 #
coedge $-1 -1 $-1 $24 $23 $22 $30 reversed $10 0 $36 #
edge $-1 5 $-1 $37 -3.141592653589793116 $37 3.141592653589793116 $29 $38 forward @7 unknown #
edge $-1 4 $-1 $34 0 $37 18.8495559215387587 $23 $39 forward @7 tangent #
pcurve $-1 -1 $-1 0 forward { exp_par_cur nubs 1 open 2 
	0 1 18.8495559215387587 1 
	3.141592653589793116 0 
	3.141592653589793116 18.8495559215387587 
	0 
	spline forward { ref 0 } I I I I 
	} 0 0 #
pcurve $-1 -1 $-1 0 forward { exp_par_cur nubs 1 open 2 
	-18.8495559215387587 1 -0 1 
	-3.141592653589793116 18.8495559215387587 
	-3.141592653589793116 0 
	0 
	spline forward { ref 0 } I I I I 
	} 0 0 #
vertex $-1 7 $-1 $26 2 $40 #
ellipse-curve $-1 -1 $-1 4.549999999999999822 1.650000000000000355 -0.500000000000000222 0 1 0 -0.04000000000000003553 0 0 1 I I #
pcurve $-1 -1 $-1 0 forward { exp_par_cur nubs 1 periodic 2 
	-3.141592653589793116 1 3.141592653589793116 1 
	3.141592653589793116 18.8495559215387587 
	-3.141592653589793116 18.8495559215387587 
	0 
	spline forward { ref 0 } I I I I 
	} 0 0 #
vertex $-1 8 $-1 $31 1 $41 #
ellipse-curve $-1 -1 $-1 4.549999999999999822 1.650000000000000133 -0.8500000000000003109 -7.34788079488411875e-16 1 0 -0.04000000000000003553 -2.939152317953649817e-17 0 1 I I #
intcurve-curve $-1 -1 $-1 forward { helix_int_cur 22140 F 0 F 18.8495559215387587 
	4.950000000000000178 1.650000000000000355 -0.500000000000000222 
	-0.3600000000000003197 0 0 
	0 0.3600000000000003197 0 
	0 0 -0.1166666666666666824 
	0 
	0 0 -1 
	null_surface 
	null_surface 
	nullbs 
	nullbs 
	} I I #
point $-1 -1 $-1 4.589999999999999858 1.650000000000000355 -0.500000000000000222 #
point $-1 -1 $-1 4.589999999999999858 1.650000000000000133 -0.8500000000000003109 #
End-of-ASM-data 