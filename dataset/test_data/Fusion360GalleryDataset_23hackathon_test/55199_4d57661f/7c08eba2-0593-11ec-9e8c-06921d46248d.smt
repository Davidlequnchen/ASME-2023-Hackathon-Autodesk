22700 0 2 4           
16 Autodesk Neutron 20 ASM 227.1.1.65535 NT 24 Wed Aug 25 11:00:20 2021 
10 9.999999999999999547e-07 1.000000000000000036e-10 
asmheader $-1 -1 @13 227.1.1.65535 #
body $2 15 $-1 $3 $-1 $4 #
ATTRIB_CUSTOM-attrib $-1 -1 $5 $-1 $1 @20 Timestamp_attrib_def 1 1469381328001 #
lump $-1 -1 $-1 $-1 $6 $1 #
transform $-1 -1 1 0 0 0 1 0 0 0 1 0 0 0 1 no_rotate no_reflect no_shear #
ATTRIB_CUSTOM-attrib $-1 -1 $-1 $2 $1 @22 generic_tag_attrib_def 3 3 -1 @23 generic_tag_attrib_def  1 3 @3 303 4 0 0 #
shell $-1 -1 $-1 $-1 $-1 $7 $-1 $3 #
face $8 10 $-1 $9 $10 $6 $-1 $11 forward single #
ATTRIB_CUSTOM-attrib $-1 -1 $-1 $-1 $7 @22 generic_tag_attrib_def 3 3 -1 @23 generic_tag_attrib_def  1 1 @1 3 0 1 303 0 #
face $12 11 $-1 $13 $14 $6 $-1 $15 forward single #
loop $-1 -1 $-1 $16 $17 $7 #
cone-surface $-1 -1 $-1 0 0 0 0 1 0 9.5 0 0 1 I I 0 1 9.5 forward I I I I #
ATTRIB_CUSTOM-attrib $-1 -1 $-1 $-1 $9 @22 generic_tag_attrib_def 3 3 -1 @23 generic_tag_attrib_def  1 1 @1 2 0 1 303 0 #
face $18 12 $-1 $-1 $19 $6 $-1 $20 reversed single #
loop $-1 -1 $-1 $-1 $21 $9 #
plane-surface $-1 -1 $-1 0 0.2000000000000000111 0 0 1 0 1 0 0 forward_v I I I I #
loop $-1 -1 $-1 $-1 $22 $7 #
coedge $-1 -1 $-1 $17 $17 $21 $23 reversed $10 0 $-1 #
ATTRIB_CUSTOM-attrib $-1 -1 $-1 $-1 $13 @22 generic_tag_attrib_def 3 3 -1 @23 generic_tag_attrib_def  1 1 @1 1 0 1 303 0 #
loop $-1 -1 $-1 $-1 $24 $13 #
plane-surface $-1 -1 $-1 0 0 0 0 1 0 1 0 0 forward_v I I I I #
coedge $-1 -1 $-1 $21 $21 $17 $23 forward $14 0 $-1 #
coedge $25 -1 $-1 $22 $22 $24 $26 forward $16 0 $-1 #
edge $-1 13 $-1 $27 -3.141592653589793116 $27 3.141592653589793116 $17 $28 forward @7 unknown #
coedge $29 -1 $-1 $24 $24 $22 $26 reversed $19 0 $-1 #
ATTRIB_CUSTOM-attrib $-1 -1 $-1 $-1 $22 @17 sketch_attrib_def 1 1 3 @13 102 0 0 0 1 1 #
edge $-1 14 $-1 $30 -3.141592653589793116 $30 3.141592653589793116 $22 $31 forward @7 unknown #
vertex $-1 51 $-1 $23 2 $32 #
ellipse-curve $-1 -1 $-1 0 0.2000000000000000111 0 0 1 0 9.5 0 0 1 I I #
ATTRIB_CUSTOM-attrib $-1 -1 $-1 $-1 $24 @17 sketch_attrib_def 1 1 3 @13 102 0 0 0 1 1 #
vertex $-1 52 $-1 $26 0 $33 #
ellipse-curve $-1 -1 $-1 0 0 0 0 1 0 9.5 0 0 1 I I #
point $-1 -1 $-1 -9.5 0.2000000000000000111 1.163414459189985485e-15 #
point $-1 -1 $-1 -9.5 0 1.163414459189985485e-15 #
End-of-ASM-data 