??
?'?'
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
K
Bincount
arr
size
weights"T	
bins"T"
Ttype:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Cumsum
x"T
axis"Tidx
out"T"
	exclusivebool( "
reversebool( " 
Ttype:
2	"
Tidxtype0:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
Tvaluestype?
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
:
Maximum
x"T
y"T
z"T"
Ttype:

2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
:
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	?
?
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
RaggedTensorToTensor
shape"Tshape
values"T
default_value"T:
row_partition_tensors"Tindex*num_row_partition_tensors
result"T"	
Ttype"
Tindextype:
2	"
Tshapetype:
2	"$
num_row_partition_tensorsint(0"#
row_partition_typeslist(string)
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
?
StatelessIf
cond"Tcond
input2Tin
output2Tout"
Tcondtype"
Tin
list(type)("
Tout
list(type)("
then_branchfunc"
else_branchfunc" 
output_shapeslist(shape)
 
@
StaticRegexFullMatch	
input

output
"
patternstring
m
StaticRegexReplace	
input

output"
patternstring"
rewritestring"
replace_globalbool(
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
StringLower	
input

output"
encodingstring 
e
StringSplitV2	
input
sep
indices	

values	
shape	"
maxsplitint?????????
;
Sub
x"T
y"T
z"T"
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8Ƶ
?
embedding_2/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameembedding_2/embeddings
?
*embedding_2/embeddings/Read/ReadVariableOpReadVariableOpembedding_2/embeddings* 
_output_shapes
:
??*
dtype0
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:*
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
dtype0
?
string_lookup_1_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_226780*
value_dtype0	
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
?
Adam/embedding_2/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*.
shared_nameAdam/embedding_2/embeddings/m
?
1Adam/embedding_2/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_2/embeddings/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_4/kernel/m

)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/m
w
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_5/kernel/m

)Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/m
w
'Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/m*
_output_shapes
:*
dtype0
?
Adam/embedding_2/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*.
shared_nameAdam/embedding_2/embeddings/v
?
1Adam/embedding_2/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_2/embeddings/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_4/kernel/v

)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/v
w
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_5/kernel/v

)Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/v
w
'Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/v*
_output_shapes
:*
dtype0
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *$
fR
__inference_<lambda>_323841

NoOpNoOp^PartitionedCall
?
Jstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2string_lookup_1_index_table*
Tkeys0*
Tvalues0	*.
_class$
" loc:@string_lookup_1_index_table*
_output_shapes

::
?5
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*?4
value?4B?4 B?4
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
=
	state_variables

_index_lookup_layer
	keras_api
?
layer_with_weights-0
layer-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api
 
#
0
1
2
3
4
 
#
1
2
3
4
5
?
trainable_variables
non_trainable_variables
layer_regularization_losses
regularization_losses
	variables
metrics
layer_metrics

 layers
 
 
0
!state_variables

"_table
#	keras_api
 
b

embeddings
$trainable_variables
%regularization_losses
&	variables
'	keras_api
R
(trainable_variables
)regularization_losses
*	variables
+	keras_api
R
,trainable_variables
-regularization_losses
.	variables
/	keras_api
R
0trainable_variables
1regularization_losses
2	variables
3	keras_api
h

kernel
bias
4trainable_variables
5regularization_losses
6	variables
7	keras_api
h

kernel
bias
8trainable_variables
9regularization_losses
:	variables
;	keras_api
?
<iter

=beta_1

>beta_2
	?decay
@learning_ratemzm{m|m}m~vv?v?v?v?
#
0
1
2
3
4
 
#
0
1
2
3
4
?
trainable_variables
Anon_trainable_variables
Blayer_regularization_losses
regularization_losses
	variables
Cmetrics
Dlayer_metrics

Elayers
\Z
VARIABLE_VALUEembedding_2/embeddings0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_4/kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_4/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_5/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_5/bias0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
 
 

F0
G1
 

0
1
 
LJ
tableAlayer_with_weights-0/_index_lookup_layer/_table/.ATTRIBUTES/table
 

0
 

0
?
$trainable_variables
Hnon_trainable_variables
Ilayer_regularization_losses
%regularization_losses
&	variables
Jmetrics
Klayer_metrics

Llayers
 
 
 
?
(trainable_variables
Mnon_trainable_variables
Nlayer_regularization_losses
)regularization_losses
*	variables
Ometrics
Player_metrics

Qlayers
 
 
 
?
,trainable_variables
Rnon_trainable_variables
Slayer_regularization_losses
-regularization_losses
.	variables
Tmetrics
Ulayer_metrics

Vlayers
 
 
 
?
0trainable_variables
Wnon_trainable_variables
Xlayer_regularization_losses
1regularization_losses
2	variables
Ymetrics
Zlayer_metrics

[layers

0
1
 

0
1
?
4trainable_variables
\non_trainable_variables
]layer_regularization_losses
5regularization_losses
6	variables
^metrics
_layer_metrics

`layers

0
1
 

0
1
?
8trainable_variables
anon_trainable_variables
blayer_regularization_losses
9regularization_losses
:	variables
cmetrics
dlayer_metrics

elayers
][
VARIABLE_VALUE	Adam/iter>layer_with_weights-1/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEAdam/beta_1@layer_with_weights-1/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEAdam/beta_2@layer_with_weights-1/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE
Adam/decay?layer_with_weights-1/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/learning_rateGlayer_with_weights-1/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 

f0
g1
 
*
0
1
2
3
4
5
4
	htotal
	icount
j	variables
k	keras_api
D
	ltotal
	mcount
n
_fn_kwargs
o	variables
p	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	qtotal
	rcount
s	variables
t	keras_api
D
	utotal
	vcount
w
_fn_kwargs
x	variables
y	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

h0
i1

j	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

l0
m1

o	variables
fd
VARIABLE_VALUEtotal_2Ilayer_with_weights-1/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEcount_2Ilayer_with_weights-1/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

q0
r1

s	variables
fd
VARIABLE_VALUEtotal_3Ilayer_with_weights-1/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEcount_3Ilayer_with_weights-1/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

u0
v1

x	variables
??
VARIABLE_VALUEAdam/embedding_2/embeddings/matrainable_variables/0/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_4/kernel/matrainable_variables/1/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_4/bias/matrainable_variables/2/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_5/kernel/matrainable_variables/3/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_5/bias/matrainable_variables/4/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/embedding_2/embeddings/vatrainable_variables/0/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_4/kernel/vatrainable_variables/1/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_4/bias/vatrainable_variables/2/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_5/kernel/vatrainable_variables/3/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_5/bias/vatrainable_variables/4/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
*serving_default_text_vectorization_1_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCall*serving_default_text_vectorization_1_inputstring_lookup_1_index_tableConstembedding_2/embeddingsdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_323162
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*embedding_2/embeddings/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOpJstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2Lstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2:1Adam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOp1Adam/embedding_2/embeddings/m/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp)Adam/dense_5/kernel/m/Read/ReadVariableOp'Adam/dense_5/bias/m/Read/ReadVariableOp1Adam/embedding_2/embeddings/v/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOp)Adam/dense_5/kernel/v/Read/ReadVariableOp'Adam/dense_5/bias/v/Read/ReadVariableOpConst_1*+
Tin$
"2 		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_323955
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_2/embeddingsdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasstring_lookup_1_index_table	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2total_3count_3Adam/embedding_2/embeddings/mAdam/dense_4/kernel/mAdam/dense_4/bias/mAdam/dense_5/kernel/mAdam/dense_5/bias/mAdam/embedding_2/embeddings/vAdam/dense_4/kernel/vAdam/dense_4/bias/vAdam/dense_5/kernel/vAdam/dense_5/bias/v*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_324052??
?
?
-__inference_sequential_4_layer_call_fn_323141
text_vectorization_1_input
unknown
	unknown_0	
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_3231242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:?????????:: :::::22
StatefulPartitionedCallStatefulPartitionedCall:c _
'
_output_shapes
:?????????
4
_user_specified_nametext_vectorization_1_input:

_output_shapes
: 
?
-
__inference__destroyer_323809
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
d
E__inference_dropout_4_layer_call_and_return_conditional_losses_323690

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const?
dropout/MulMulinputsdropout/Const:output:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/Mul_1r
IdentityIdentitydropout/Mul_1:z:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?#
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_323531

inputs	'
#embedding_2_embedding_lookup_323507*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identity??dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?embedding_2/embedding_lookup]
CastCastinputs*

DstT0*

SrcT0	*'
_output_shapes
:?????????d2
Castw
embedding_2/CastCastCast:y:0*

DstT0*

SrcT0*'
_output_shapes
:?????????d2
embedding_2/Cast?
embedding_2/embedding_lookupResourceGather#embedding_2_embedding_lookup_323507embedding_2/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_2/embedding_lookup/323507*+
_output_shapes
:?????????d*
dtype02
embedding_2/embedding_lookup?
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_2/embedding_lookup/323507*+
_output_shapes
:?????????d2'
%embedding_2/embedding_lookup/Identity?
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????d2)
'embedding_2/embedding_lookup/Identity_1?
dropout_4/IdentityIdentity0embedding_2/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????d2
dropout_4/Identity?
1global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_2/Mean/reduction_indices?
global_average_pooling1d_2/MeanMeandropout_4/Identity:output:0:global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2!
global_average_pooling1d_2/Mean?
dropout_5/IdentityIdentity(global_average_pooling1d_2/Mean:output:0*
T0*'
_output_shapes
:?????????2
dropout_5/Identity?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMuldropout_5/Identity:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_4/Relu?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/BiasAddy
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_5/Sigmoid?
IdentityIdentitydense_5/Sigmoid:y:0^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^embedding_2/embedding_lookup*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????d:::::2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2<
embedding_2/embedding_lookupembedding_2/embedding_lookup:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
-__inference_sequential_3_layer_call_fn_322635
embedding_2_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallembedding_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_3226222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:c _
0
_output_shapes
:??????????????????
+
_user_specified_nameembedding_2_input
??
?
H__inference_sequential_4_layer_call_and_return_conditional_losses_322829
text_vectorization_1_input^
Ztext_vectorization_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handle_
[text_vectorization_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value	
sequential_3_322817
sequential_3_322819
sequential_3_322821
sequential_3_322823
sequential_3_322825
identity??$sequential_3/StatefulPartitionedCall?Mtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2?
 text_vectorization_1/StringLowerStringLowertext_vectorization_1_input*'
_output_shapes
:?????????2"
 text_vectorization_1/StringLower?
'text_vectorization_1/StaticRegexReplaceStaticRegexReplace)text_vectorization_1/StringLower:output:0*'
_output_shapes
:?????????*
pattern<br />*
rewrite 2)
'text_vectorization_1/StaticRegexReplace?
)text_vectorization_1/StaticRegexReplace_1StaticRegexReplace0text_vectorization_1/StaticRegexReplace:output:0*'
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite 2+
)text_vectorization_1/StaticRegexReplace_1?
text_vectorization_1/SqueezeSqueeze2text_vectorization_1/StaticRegexReplace_1:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????2
text_vectorization_1/Squeeze?
&text_vectorization_1/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2(
&text_vectorization_1/StringSplit/Const?
.text_vectorization_1/StringSplit/StringSplitV2StringSplitV2%text_vectorization_1/Squeeze:output:0/text_vectorization_1/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:20
.text_vectorization_1/StringSplit/StringSplitV2?
4text_vectorization_1/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        26
4text_vectorization_1/StringSplit/strided_slice/stack?
6text_vectorization_1/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       28
6text_vectorization_1/StringSplit/strided_slice/stack_1?
6text_vectorization_1/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      28
6text_vectorization_1/StringSplit/strided_slice/stack_2?
.text_vectorization_1/StringSplit/strided_sliceStridedSlice8text_vectorization_1/StringSplit/StringSplitV2:indices:0=text_vectorization_1/StringSplit/strided_slice/stack:output:0?text_vectorization_1/StringSplit/strided_slice/stack_1:output:0?text_vectorization_1/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask20
.text_vectorization_1/StringSplit/strided_slice?
6text_vectorization_1/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6text_vectorization_1/StringSplit/strided_slice_1/stack?
8text_vectorization_1/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8text_vectorization_1/StringSplit/strided_slice_1/stack_1?
8text_vectorization_1/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8text_vectorization_1/StringSplit/strided_slice_1/stack_2?
0text_vectorization_1/StringSplit/strided_slice_1StridedSlice6text_vectorization_1/StringSplit/StringSplitV2:shape:0?text_vectorization_1/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_1/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_1/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask22
0text_vectorization_1/StringSplit/strided_slice_1?
Wtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_1/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2Y
Wtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast?
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_1/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2[
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1?
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2c
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape?
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2c
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const?
`text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2b
`text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod?
etext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2g
etext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y?
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2e
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater?
`text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2b
`text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast?
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2e
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1?
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2a
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max?
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2c
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y?
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2a
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add?
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2a
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul?
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2e
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum?
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2e
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum?
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2e
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2?
dtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:?????????2f
dtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount?
^text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2`
^text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis?
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:?????????2[
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum?
btext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2d
btext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0?
^text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2`
^text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis?
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:?????????2[
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat?
Mtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Ztext_vectorization_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handle7text_vectorization_1/StringSplit/StringSplitV2:values:0[text_vectorization_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:?????????2O
Mtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2?
6text_vectorization_1/string_lookup_1/assert_equal/NoOpNoOp*
_output_shapes
 28
6text_vectorization_1/string_lookup_1/assert_equal/NoOp?
-text_vectorization_1/string_lookup_1/IdentityIdentityVtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2/
-text_vectorization_1/string_lookup_1/Identity?
/text_vectorization_1/string_lookup_1/Identity_1Identitybtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:?????????21
/text_vectorization_1/string_lookup_1/Identity_1?
1text_vectorization_1/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 23
1text_vectorization_1/RaggedToTensor/default_value?
)text_vectorization_1/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2+
)text_vectorization_1/RaggedToTensor/Const?
8text_vectorization_1/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_1/RaggedToTensor/Const:output:06text_vectorization_1/string_lookup_1/Identity:output:0:text_vectorization_1/RaggedToTensor/default_value:output:08text_vectorization_1/string_lookup_1/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:??????????????????*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2:
8text_vectorization_1/RaggedToTensor/RaggedTensorToTensor?
text_vectorization_1/ShapeShapeAtext_vectorization_1/RaggedToTensor/RaggedTensorToTensor:result:0*
T0	*
_output_shapes
:2
text_vectorization_1/Shape?
(text_vectorization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(text_vectorization_1/strided_slice/stack?
*text_vectorization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*text_vectorization_1/strided_slice/stack_1?
*text_vectorization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*text_vectorization_1/strided_slice/stack_2?
"text_vectorization_1/strided_sliceStridedSlice#text_vectorization_1/Shape:output:01text_vectorization_1/strided_slice/stack:output:03text_vectorization_1/strided_slice/stack_1:output:03text_vectorization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"text_vectorization_1/strided_slicez
text_vectorization_1/sub/xConst*
_output_shapes
: *
dtype0*
value	B :d2
text_vectorization_1/sub/x?
text_vectorization_1/subSub#text_vectorization_1/sub/x:output:0+text_vectorization_1/strided_slice:output:0*
T0*
_output_shapes
: 2
text_vectorization_1/sub|
text_vectorization_1/Less/yConst*
_output_shapes
: *
dtype0*
value	B :d2
text_vectorization_1/Less/y?
text_vectorization_1/LessLess+text_vectorization_1/strided_slice:output:0$text_vectorization_1/Less/y:output:0*
T0*
_output_shapes
: 2
text_vectorization_1/Less?
text_vectorization_1/condStatelessIftext_vectorization_1/Less:z:0text_vectorization_1/sub:z:0Atext_vectorization_1/RaggedToTensor/RaggedTensorToTensor:result:0*
Tcond0
*
Tin
2	*
Tout
2	*
_lower_using_switch_merge(*0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *9
else_branch*R(
&text_vectorization_1_cond_false_322695*/
output_shapes
:??????????????????*8
then_branch)R'
%text_vectorization_1_cond_true_3226942
text_vectorization_1/cond?
"text_vectorization_1/cond/IdentityIdentity"text_vectorization_1/cond:output:0*
T0	*'
_output_shapes
:?????????d2$
"text_vectorization_1/cond/Identity?
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall+text_vectorization_1/cond/Identity:output:0sequential_3_322817sequential_3_322819sequential_3_322821sequential_3_322823sequential_3_322825*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_3227572&
$sequential_3/StatefulPartitionedCall?
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:0%^sequential_3/StatefulPartitionedCallN^text_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:?????????:: :::::2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall2?
Mtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2Mtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2:c _
'
_output_shapes
:?????????
4
_user_specified_nametext_vectorization_1_input:

_output_shapes
: 
?#
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_322786

inputs	'
#embedding_2_embedding_lookup_322762*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identity??dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?embedding_2/embedding_lookup]
CastCastinputs*

DstT0*

SrcT0	*'
_output_shapes
:?????????d2
Castw
embedding_2/CastCastCast:y:0*

DstT0*

SrcT0*'
_output_shapes
:?????????d2
embedding_2/Cast?
embedding_2/embedding_lookupResourceGather#embedding_2_embedding_lookup_322762embedding_2/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_2/embedding_lookup/322762*+
_output_shapes
:?????????d*
dtype02
embedding_2/embedding_lookup?
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_2/embedding_lookup/322762*+
_output_shapes
:?????????d2'
%embedding_2/embedding_lookup/Identity?
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????d2)
'embedding_2/embedding_lookup/Identity_1?
dropout_4/IdentityIdentity0embedding_2/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????d2
dropout_4/Identity?
1global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_2/Mean/reduction_indices?
global_average_pooling1d_2/MeanMeandropout_4/Identity:output:0:global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2!
global_average_pooling1d_2/Mean?
dropout_5/IdentityIdentity(global_average_pooling1d_2/Mean:output:0*
T0*'
_output_shapes
:?????????2
dropout_5/Identity?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMuldropout_5/Identity:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_4/Relu?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/BiasAddy
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_5/Sigmoid?
IdentityIdentitydense_5/Sigmoid:y:0^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^embedding_2/embedding_lookup*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????d:::::2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2<
embedding_2/embedding_lookupembedding_2/embedding_lookup:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
r
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_322388

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?	
?
C__inference_dense_5_layer_call_and_return_conditional_losses_322527

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
!__inference__wrapped_model_322372
text_vectorization_1_inputk
gsequential_4_text_vectorization_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handlel
hsequential_4_text_vectorization_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value	A
=sequential_4_sequential_3_embedding_2_embedding_lookup_322348D
@sequential_4_sequential_3_dense_4_matmul_readvariableop_resourceE
Asequential_4_sequential_3_dense_4_biasadd_readvariableop_resourceD
@sequential_4_sequential_3_dense_5_matmul_readvariableop_resourceE
Asequential_4_sequential_3_dense_5_biasadd_readvariableop_resource
identity??8sequential_4/sequential_3/dense_4/BiasAdd/ReadVariableOp?7sequential_4/sequential_3/dense_4/MatMul/ReadVariableOp?8sequential_4/sequential_3/dense_5/BiasAdd/ReadVariableOp?7sequential_4/sequential_3/dense_5/MatMul/ReadVariableOp?6sequential_4/sequential_3/embedding_2/embedding_lookup?Zsequential_4/text_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2?
-sequential_4/text_vectorization_1/StringLowerStringLowertext_vectorization_1_input*'
_output_shapes
:?????????2/
-sequential_4/text_vectorization_1/StringLower?
4sequential_4/text_vectorization_1/StaticRegexReplaceStaticRegexReplace6sequential_4/text_vectorization_1/StringLower:output:0*'
_output_shapes
:?????????*
pattern<br />*
rewrite 26
4sequential_4/text_vectorization_1/StaticRegexReplace?
6sequential_4/text_vectorization_1/StaticRegexReplace_1StaticRegexReplace=sequential_4/text_vectorization_1/StaticRegexReplace:output:0*'
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite 28
6sequential_4/text_vectorization_1/StaticRegexReplace_1?
)sequential_4/text_vectorization_1/SqueezeSqueeze?sequential_4/text_vectorization_1/StaticRegexReplace_1:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????2+
)sequential_4/text_vectorization_1/Squeeze?
3sequential_4/text_vectorization_1/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 25
3sequential_4/text_vectorization_1/StringSplit/Const?
;sequential_4/text_vectorization_1/StringSplit/StringSplitV2StringSplitV22sequential_4/text_vectorization_1/Squeeze:output:0<sequential_4/text_vectorization_1/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:2=
;sequential_4/text_vectorization_1/StringSplit/StringSplitV2?
Asequential_4/text_vectorization_1/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2C
Asequential_4/text_vectorization_1/StringSplit/strided_slice/stack?
Csequential_4/text_vectorization_1/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2E
Csequential_4/text_vectorization_1/StringSplit/strided_slice/stack_1?
Csequential_4/text_vectorization_1/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2E
Csequential_4/text_vectorization_1/StringSplit/strided_slice/stack_2?
;sequential_4/text_vectorization_1/StringSplit/strided_sliceStridedSliceEsequential_4/text_vectorization_1/StringSplit/StringSplitV2:indices:0Jsequential_4/text_vectorization_1/StringSplit/strided_slice/stack:output:0Lsequential_4/text_vectorization_1/StringSplit/strided_slice/stack_1:output:0Lsequential_4/text_vectorization_1/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2=
;sequential_4/text_vectorization_1/StringSplit/strided_slice?
Csequential_4/text_vectorization_1/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2E
Csequential_4/text_vectorization_1/StringSplit/strided_slice_1/stack?
Esequential_4/text_vectorization_1/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Esequential_4/text_vectorization_1/StringSplit/strided_slice_1/stack_1?
Esequential_4/text_vectorization_1/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Esequential_4/text_vectorization_1/StringSplit/strided_slice_1/stack_2?
=sequential_4/text_vectorization_1/StringSplit/strided_slice_1StridedSliceCsequential_4/text_vectorization_1/StringSplit/StringSplitV2:shape:0Lsequential_4/text_vectorization_1/StringSplit/strided_slice_1/stack:output:0Nsequential_4/text_vectorization_1/StringSplit/strided_slice_1/stack_1:output:0Nsequential_4/text_vectorization_1/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2?
=sequential_4/text_vectorization_1/StringSplit/strided_slice_1?
dsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCastDsequential_4/text_vectorization_1/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2f
dsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast?
fsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1CastFsequential_4/text_vectorization_1/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2h
fsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1?
nsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapehsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2p
nsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape?
nsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2p
nsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const?
msequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdwsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0wsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2o
msequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod?
rsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2t
rsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y?
psequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatervsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0{sequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2r
psequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater?
msequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCasttsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2o
msequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast?
psequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2r
psequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1?
lsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxhsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ysequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2n
lsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max?
nsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2p
nsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y?
lsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2usequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0wsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2n
lsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add?
lsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulqsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0psequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2n
lsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul?
psequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumjsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0psequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2r
psequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum?
psequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumjsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0tsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2r
psequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum?
psequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2r
psequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2?
qsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincounthsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0tsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ysequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:?????????2s
qsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount?
ksequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2m
ksequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis?
fsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumxsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0tsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:?????????2h
fsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum?
osequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2q
osequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0?
ksequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2m
ksequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis?
fsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2xsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0lsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0tsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:?????????2h
fsequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat?
Zsequential_4/text_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2LookupTableFindV2gsequential_4_text_vectorization_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handleDsequential_4/text_vectorization_1/StringSplit/StringSplitV2:values:0hsequential_4_text_vectorization_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:?????????2\
Zsequential_4/text_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2?
Csequential_4/text_vectorization_1/string_lookup_1/assert_equal/NoOpNoOp*
_output_shapes
 2E
Csequential_4/text_vectorization_1/string_lookup_1/assert_equal/NoOp?
:sequential_4/text_vectorization_1/string_lookup_1/IdentityIdentitycsequential_4/text_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2<
:sequential_4/text_vectorization_1/string_lookup_1/Identity?
<sequential_4/text_vectorization_1/string_lookup_1/Identity_1Identityosequential_4/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:?????????2>
<sequential_4/text_vectorization_1/string_lookup_1/Identity_1?
>sequential_4/text_vectorization_1/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 2@
>sequential_4/text_vectorization_1/RaggedToTensor/default_value?
6sequential_4/text_vectorization_1/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????28
6sequential_4/text_vectorization_1/RaggedToTensor/Const?
Esequential_4/text_vectorization_1/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor?sequential_4/text_vectorization_1/RaggedToTensor/Const:output:0Csequential_4/text_vectorization_1/string_lookup_1/Identity:output:0Gsequential_4/text_vectorization_1/RaggedToTensor/default_value:output:0Esequential_4/text_vectorization_1/string_lookup_1/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:??????????????????*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2G
Esequential_4/text_vectorization_1/RaggedToTensor/RaggedTensorToTensor?
'sequential_4/text_vectorization_1/ShapeShapeNsequential_4/text_vectorization_1/RaggedToTensor/RaggedTensorToTensor:result:0*
T0	*
_output_shapes
:2)
'sequential_4/text_vectorization_1/Shape?
5sequential_4/text_vectorization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5sequential_4/text_vectorization_1/strided_slice/stack?
7sequential_4/text_vectorization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_4/text_vectorization_1/strided_slice/stack_1?
7sequential_4/text_vectorization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_4/text_vectorization_1/strided_slice/stack_2?
/sequential_4/text_vectorization_1/strided_sliceStridedSlice0sequential_4/text_vectorization_1/Shape:output:0>sequential_4/text_vectorization_1/strided_slice/stack:output:0@sequential_4/text_vectorization_1/strided_slice/stack_1:output:0@sequential_4/text_vectorization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_4/text_vectorization_1/strided_slice?
'sequential_4/text_vectorization_1/sub/xConst*
_output_shapes
: *
dtype0*
value	B :d2)
'sequential_4/text_vectorization_1/sub/x?
%sequential_4/text_vectorization_1/subSub0sequential_4/text_vectorization_1/sub/x:output:08sequential_4/text_vectorization_1/strided_slice:output:0*
T0*
_output_shapes
: 2'
%sequential_4/text_vectorization_1/sub?
(sequential_4/text_vectorization_1/Less/yConst*
_output_shapes
: *
dtype0*
value	B :d2*
(sequential_4/text_vectorization_1/Less/y?
&sequential_4/text_vectorization_1/LessLess8sequential_4/text_vectorization_1/strided_slice:output:01sequential_4/text_vectorization_1/Less/y:output:0*
T0*
_output_shapes
: 2(
&sequential_4/text_vectorization_1/Less?
&sequential_4/text_vectorization_1/condStatelessIf*sequential_4/text_vectorization_1/Less:z:0)sequential_4/text_vectorization_1/sub:z:0Nsequential_4/text_vectorization_1/RaggedToTensor/RaggedTensorToTensor:result:0*
Tcond0
*
Tin
2	*
Tout
2	*
_lower_using_switch_merge(*0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *F
else_branch7R5
3sequential_4_text_vectorization_1_cond_false_322326*/
output_shapes
:??????????????????*E
then_branch6R4
2sequential_4_text_vectorization_1_cond_true_3223252(
&sequential_4/text_vectorization_1/cond?
/sequential_4/text_vectorization_1/cond/IdentityIdentity/sequential_4/text_vectorization_1/cond:output:0*
T0	*'
_output_shapes
:?????????d21
/sequential_4/text_vectorization_1/cond/Identity?
sequential_4/sequential_3/CastCast8sequential_4/text_vectorization_1/cond/Identity:output:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????d2 
sequential_4/sequential_3/Cast?
*sequential_4/sequential_3/embedding_2/CastCast"sequential_4/sequential_3/Cast:y:0*

DstT0*

SrcT0*'
_output_shapes
:?????????d2,
*sequential_4/sequential_3/embedding_2/Cast?
6sequential_4/sequential_3/embedding_2/embedding_lookupResourceGather=sequential_4_sequential_3_embedding_2_embedding_lookup_322348.sequential_4/sequential_3/embedding_2/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*P
_classF
DBloc:@sequential_4/sequential_3/embedding_2/embedding_lookup/322348*+
_output_shapes
:?????????d*
dtype028
6sequential_4/sequential_3/embedding_2/embedding_lookup?
?sequential_4/sequential_3/embedding_2/embedding_lookup/IdentityIdentity?sequential_4/sequential_3/embedding_2/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*P
_classF
DBloc:@sequential_4/sequential_3/embedding_2/embedding_lookup/322348*+
_output_shapes
:?????????d2A
?sequential_4/sequential_3/embedding_2/embedding_lookup/Identity?
Asequential_4/sequential_3/embedding_2/embedding_lookup/Identity_1IdentityHsequential_4/sequential_3/embedding_2/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????d2C
Asequential_4/sequential_3/embedding_2/embedding_lookup/Identity_1?
,sequential_4/sequential_3/dropout_4/IdentityIdentityJsequential_4/sequential_3/embedding_2/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????d2.
,sequential_4/sequential_3/dropout_4/Identity?
Ksequential_4/sequential_3/global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2M
Ksequential_4/sequential_3/global_average_pooling1d_2/Mean/reduction_indices?
9sequential_4/sequential_3/global_average_pooling1d_2/MeanMean5sequential_4/sequential_3/dropout_4/Identity:output:0Tsequential_4/sequential_3/global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2;
9sequential_4/sequential_3/global_average_pooling1d_2/Mean?
,sequential_4/sequential_3/dropout_5/IdentityIdentityBsequential_4/sequential_3/global_average_pooling1d_2/Mean:output:0*
T0*'
_output_shapes
:?????????2.
,sequential_4/sequential_3/dropout_5/Identity?
7sequential_4/sequential_3/dense_4/MatMul/ReadVariableOpReadVariableOp@sequential_4_sequential_3_dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype029
7sequential_4/sequential_3/dense_4/MatMul/ReadVariableOp?
(sequential_4/sequential_3/dense_4/MatMulMatMul5sequential_4/sequential_3/dropout_5/Identity:output:0?sequential_4/sequential_3/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2*
(sequential_4/sequential_3/dense_4/MatMul?
8sequential_4/sequential_3/dense_4/BiasAdd/ReadVariableOpReadVariableOpAsequential_4_sequential_3_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8sequential_4/sequential_3/dense_4/BiasAdd/ReadVariableOp?
)sequential_4/sequential_3/dense_4/BiasAddBiasAdd2sequential_4/sequential_3/dense_4/MatMul:product:0@sequential_4/sequential_3/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2+
)sequential_4/sequential_3/dense_4/BiasAdd?
&sequential_4/sequential_3/dense_4/ReluRelu2sequential_4/sequential_3/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2(
&sequential_4/sequential_3/dense_4/Relu?
7sequential_4/sequential_3/dense_5/MatMul/ReadVariableOpReadVariableOp@sequential_4_sequential_3_dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype029
7sequential_4/sequential_3/dense_5/MatMul/ReadVariableOp?
(sequential_4/sequential_3/dense_5/MatMulMatMul4sequential_4/sequential_3/dense_4/Relu:activations:0?sequential_4/sequential_3/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2*
(sequential_4/sequential_3/dense_5/MatMul?
8sequential_4/sequential_3/dense_5/BiasAdd/ReadVariableOpReadVariableOpAsequential_4_sequential_3_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8sequential_4/sequential_3/dense_5/BiasAdd/ReadVariableOp?
)sequential_4/sequential_3/dense_5/BiasAddBiasAdd2sequential_4/sequential_3/dense_5/MatMul:product:0@sequential_4/sequential_3/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2+
)sequential_4/sequential_3/dense_5/BiasAdd?
)sequential_4/sequential_3/dense_5/SigmoidSigmoid2sequential_4/sequential_3/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2+
)sequential_4/sequential_3/dense_5/Sigmoid?
IdentityIdentity-sequential_4/sequential_3/dense_5/Sigmoid:y:09^sequential_4/sequential_3/dense_4/BiasAdd/ReadVariableOp8^sequential_4/sequential_3/dense_4/MatMul/ReadVariableOp9^sequential_4/sequential_3/dense_5/BiasAdd/ReadVariableOp8^sequential_4/sequential_3/dense_5/MatMul/ReadVariableOp7^sequential_4/sequential_3/embedding_2/embedding_lookup[^sequential_4/text_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:?????????:: :::::2t
8sequential_4/sequential_3/dense_4/BiasAdd/ReadVariableOp8sequential_4/sequential_3/dense_4/BiasAdd/ReadVariableOp2r
7sequential_4/sequential_3/dense_4/MatMul/ReadVariableOp7sequential_4/sequential_3/dense_4/MatMul/ReadVariableOp2t
8sequential_4/sequential_3/dense_5/BiasAdd/ReadVariableOp8sequential_4/sequential_3/dense_5/BiasAdd/ReadVariableOp2r
7sequential_4/sequential_3/dense_5/MatMul/ReadVariableOp7sequential_4/sequential_3/dense_5/MatMul/ReadVariableOp2p
6sequential_4/sequential_3/embedding_2/embedding_lookup6sequential_4/sequential_3/embedding_2/embedding_lookup2?
Zsequential_4/text_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2Zsequential_4/text_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2:c _
'
_output_shapes
:?????????
4
_user_specified_nametext_vectorization_1_input:

_output_shapes
: 
?
c
E__inference_dropout_5_layer_call_and_return_conditional_losses_323744

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_4_layer_call_fn_323705

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_3224342
PartitionedCally
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
&text_vectorization_1_cond_false_323248)
%text_vectorization_1_cond_placeholderd
`text_vectorization_1_cond_strided_slice_text_vectorization_1_raggedtotensor_raggedtensortotensor	&
"text_vectorization_1_cond_identity	?
-text_vectorization_1/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2/
-text_vectorization_1/cond/strided_slice/stack?
/text_vectorization_1/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   21
/text_vectorization_1/cond/strided_slice/stack_1?
/text_vectorization_1/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/text_vectorization_1/cond/strided_slice/stack_2?
'text_vectorization_1/cond/strided_sliceStridedSlice`text_vectorization_1_cond_strided_slice_text_vectorization_1_raggedtotensor_raggedtensortotensor6text_vectorization_1/cond/strided_slice/stack:output:08text_vectorization_1/cond/strided_slice/stack_1:output:08text_vectorization_1/cond/strided_slice/stack_2:output:0*
Index0*
T0	*0
_output_shapes
:??????????????????*

begin_mask*
end_mask2)
'text_vectorization_1/cond/strided_slice?
"text_vectorization_1/cond/IdentityIdentity0text_vectorization_1/cond/strided_slice:output:0*
T0	*0
_output_shapes
:??????????????????2$
"text_vectorization_1/cond/Identity"Q
"text_vectorization_1_cond_identity+text_vectorization_1/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?
?
&text_vectorization_1_cond_false_322982)
%text_vectorization_1_cond_placeholderd
`text_vectorization_1_cond_strided_slice_text_vectorization_1_raggedtotensor_raggedtensortotensor	&
"text_vectorization_1_cond_identity	?
-text_vectorization_1/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2/
-text_vectorization_1/cond/strided_slice/stack?
/text_vectorization_1/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   21
/text_vectorization_1/cond/strided_slice/stack_1?
/text_vectorization_1/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/text_vectorization_1/cond/strided_slice/stack_2?
'text_vectorization_1/cond/strided_sliceStridedSlice`text_vectorization_1_cond_strided_slice_text_vectorization_1_raggedtotensor_raggedtensortotensor6text_vectorization_1/cond/strided_slice/stack:output:08text_vectorization_1/cond/strided_slice/stack_1:output:08text_vectorization_1/cond/strided_slice/stack_2:output:0*
Index0*
T0	*0
_output_shapes
:??????????????????*

begin_mask*
end_mask2)
'text_vectorization_1/cond/strided_slice?
"text_vectorization_1/cond/IdentityIdentity0text_vectorization_1/cond/strided_slice:output:0*
T0	*0
_output_shapes
:??????????????????2$
"text_vectorization_1/cond/Identity"Q
"text_vectorization_1_cond_identity+text_vectorization_1/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?
?
&text_vectorization_1_cond_false_322888)
%text_vectorization_1_cond_placeholderd
`text_vectorization_1_cond_strided_slice_text_vectorization_1_raggedtotensor_raggedtensortotensor	&
"text_vectorization_1_cond_identity	?
-text_vectorization_1/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2/
-text_vectorization_1/cond/strided_slice/stack?
/text_vectorization_1/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   21
/text_vectorization_1/cond/strided_slice/stack_1?
/text_vectorization_1/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/text_vectorization_1/cond/strided_slice/stack_2?
'text_vectorization_1/cond/strided_sliceStridedSlice`text_vectorization_1_cond_strided_slice_text_vectorization_1_raggedtotensor_raggedtensortotensor6text_vectorization_1/cond/strided_slice/stack:output:08text_vectorization_1/cond/strided_slice/stack_1:output:08text_vectorization_1/cond/strided_slice/stack_2:output:0*
Index0*
T0	*0
_output_shapes
:??????????????????*

begin_mask*
end_mask2)
'text_vectorization_1/cond/strided_slice?
"text_vectorization_1/cond/IdentityIdentity0text_vectorization_1/cond/strided_slice:output:0*
T0	*0
_output_shapes
:??????????????????2$
"text_vectorization_1/cond/Identity"Q
"text_vectorization_1_cond_identity+text_vectorization_1/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?	
?
__inference_restore_fn_323836
restored_tensors_0
restored_tensors_1	N
Jstring_lookup_1_index_table_table_restore_lookuptableimportv2_table_handle
identity??=string_lookup_1_index_table_table_restore/LookupTableImportV2?
=string_lookup_1_index_table_table_restore/LookupTableImportV2LookupTableImportV2Jstring_lookup_1_index_table_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 2?
=string_lookup_1_index_table_table_restore/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Const?
IdentityIdentityConst:output:0>^string_lookup_1_index_table_table_restore/LookupTableImportV2*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:::2~
=string_lookup_1_index_table_table_restore/LookupTableImportV2=string_lookup_1_index_table_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
?	
?
C__inference_dense_4_layer_call_and_return_conditional_losses_322500

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_4_layer_call_and_return_conditional_losses_323695

inputs

identity_1g
IdentityIdentityinputs*
T0*4
_output_shapes"
 :??????????????????2

Identityv

Identity_1IdentityIdentity:output:0*
T0*4
_output_shapes"
 :??????????????????2

Identity_1"!

identity_1Identity_1:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
r
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_322452

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_5_layer_call_and_return_conditional_losses_322471

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
-__inference_sequential_3_layer_call_fn_322600
embedding_2_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallembedding_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_3225872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:c _
0
_output_shapes
:??????????????????
+
_user_specified_nameembedding_2_input
?6
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_322757

inputs	'
#embedding_2_embedding_lookup_322719*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identity??dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?embedding_2/embedding_lookup]
CastCastinputs*

DstT0*

SrcT0	*'
_output_shapes
:?????????d2
Castw
embedding_2/CastCastCast:y:0*

DstT0*

SrcT0*'
_output_shapes
:?????????d2
embedding_2/Cast?
embedding_2/embedding_lookupResourceGather#embedding_2_embedding_lookup_322719embedding_2/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_2/embedding_lookup/322719*+
_output_shapes
:?????????d*
dtype02
embedding_2/embedding_lookup?
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_2/embedding_lookup/322719*+
_output_shapes
:?????????d2'
%embedding_2/embedding_lookup/Identity?
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????d2)
'embedding_2/embedding_lookup/Identity_1w
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_4/dropout/Const?
dropout_4/dropout/MulMul0embedding_2/embedding_lookup/Identity_1:output:0 dropout_4/dropout/Const:output:0*
T0*+
_output_shapes
:?????????d2
dropout_4/dropout/Mul?
dropout_4/dropout/ShapeShape0embedding_2/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
dropout_4/dropout/Shape?
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????d*
dtype020
.dropout_4/dropout/random_uniform/RandomUniform?
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 dropout_4/dropout/GreaterEqual/y?
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????d2 
dropout_4/dropout/GreaterEqual?
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????d2
dropout_4/dropout/Cast?
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????d2
dropout_4/dropout/Mul_1?
1global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_2/Mean/reduction_indices?
global_average_pooling1d_2/MeanMeandropout_4/dropout/Mul_1:z:0:global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2!
global_average_pooling1d_2/Meanw
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_5/dropout/Const?
dropout_5/dropout/MulMul(global_average_pooling1d_2/Mean:output:0 dropout_5/dropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout_5/dropout/Mul?
dropout_5/dropout/ShapeShape(global_average_pooling1d_2/Mean:output:0*
T0*
_output_shapes
:2
dropout_5/dropout/Shape?
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype020
.dropout_5/dropout/random_uniform/RandomUniform?
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 dropout_5/dropout/GreaterEqual/y?
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2 
dropout_5/dropout/GreaterEqual?
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout_5/dropout/Cast?
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout_5/dropout/Mul_1?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMuldropout_5/dropout/Mul_1:z:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_4/Relu?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/BiasAddy
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_5/Sigmoid?
IdentityIdentitydense_5/Sigmoid:y:0^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^embedding_2/embedding_lookup*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????d:::::2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2<
embedding_2/embedding_lookupembedding_2/embedding_lookup:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
W
;__inference_global_average_pooling1d_2_layer_call_fn_323727

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *_
fZRX
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_3223882
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
}
(__inference_dense_5_layer_call_fn_323794

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_3225272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
C__inference_dense_5_layer_call_and_return_conditional_losses_323785

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
-__inference_sequential_3_layer_call_fn_323561

inputs	
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_3227862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????d:::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
c
*__inference_dropout_4_layer_call_fn_323700

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_3224292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_5_layer_call_and_return_conditional_losses_322476

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_323162
text_vectorization_1_input
unknown
	unknown_0	
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_3223722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:?????????:: :::::22
StatefulPartitionedCallStatefulPartitionedCall:c _
'
_output_shapes
:?????????
4
_user_specified_nametext_vectorization_1_input:

_output_shapes
: 
?
?
-__inference_sequential_4_layer_call_fn_323451

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_3231242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:?????????:: :::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
r
,__inference_embedding_2_layer_call_fn_323678

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_2_layer_call_and_return_conditional_losses_3224052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
??
?
"__inference__traced_restore_324052
file_prefix+
'assignvariableop_embedding_2_embeddings%
!assignvariableop_1_dense_4_kernel#
assignvariableop_2_dense_4_bias%
!assignvariableop_3_dense_5_kernel#
assignvariableop_4_dense_5_bias]
Ystring_lookup_1_index_table_table_restore_lookuptableimportv2_string_lookup_1_index_table 
assignvariableop_5_adam_iter"
assignvariableop_6_adam_beta_1"
assignvariableop_7_adam_beta_2!
assignvariableop_8_adam_decay)
%assignvariableop_9_adam_learning_rate
assignvariableop_10_total
assignvariableop_11_count
assignvariableop_12_total_1
assignvariableop_13_count_1
assignvariableop_14_total_2
assignvariableop_15_count_2
assignvariableop_16_total_3
assignvariableop_17_count_35
1assignvariableop_18_adam_embedding_2_embeddings_m-
)assignvariableop_19_adam_dense_4_kernel_m+
'assignvariableop_20_adam_dense_4_bias_m-
)assignvariableop_21_adam_dense_5_kernel_m+
'assignvariableop_22_adam_dense_5_bias_m5
1assignvariableop_23_adam_embedding_2_embeddings_v-
)assignvariableop_24_adam_dense_4_kernel_v+
'assignvariableop_25_adam_dense_4_bias_v-
)assignvariableop_26_adam_dense_5_kernel_v+
'assignvariableop_27_adam_dense_5_bias_v
identity_29??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?=string_lookup_1_index_table_table_restore/LookupTableImportV2?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_index_lookup_layer/_table/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_index_lookup_layer/_table/.ATTRIBUTES/table-valuesB>layer_with_weights-1/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-1/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/0/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/1/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/2/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/3/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/4/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/0/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/1/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/2/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/3/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/4/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes~
|:::::::::::::::::::::::::::::::*-
dtypes#
!2		2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp'assignvariableop_embedding_2_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_4_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_4_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_5_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_5_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4?
=string_lookup_1_index_table_table_restore/LookupTableImportV2LookupTableImportV2Ystring_lookup_1_index_table_table_restore_lookuptableimportv2_string_lookup_1_index_tableRestoreV2:tensors:5RestoreV2:tensors:6*	
Tin0*

Tout0	*.
_class$
" loc:@string_lookup_1_index_table*
_output_shapes
 2?
=string_lookup_1_index_table_table_restore/LookupTableImportV2k

Identity_5IdentityRestoreV2:tensors:7"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7l

Identity_8IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8l

Identity_9IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp%assignvariableop_9_adam_learning_rateIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_2Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_total_3Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_3Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp1assignvariableop_18_adam_embedding_2_embeddings_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_dense_4_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_dense_4_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_5_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_5_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp1assignvariableop_23_adam_embedding_2_embeddings_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_4_kernel_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_dense_4_bias_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_dense_5_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_dense_5_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_279
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_28Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp>^string_lookup_1_index_table_table_restore/LookupTableImportV2"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_28?
Identity_29IdentityIdentity_28:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9>^string_lookup_1_index_table_table_restore/LookupTableImportV2*
T0*
_output_shapes
: 2
Identity_29"#
identity_29Identity_29:output:0*?
_input_shapesx
v: :::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92~
=string_lookup_1_index_table_table_restore/LookupTableImportV2=string_lookup_1_index_table_table_restore/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:40
.
_class$
" loc:@string_lookup_1_index_table
??
?
H__inference_sequential_4_layer_call_and_return_conditional_losses_323413

inputs^
Ztext_vectorization_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handle_
[text_vectorization_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value	4
0sequential_3_embedding_2_embedding_lookup_3233897
3sequential_3_dense_4_matmul_readvariableop_resource8
4sequential_3_dense_4_biasadd_readvariableop_resource7
3sequential_3_dense_5_matmul_readvariableop_resource8
4sequential_3_dense_5_biasadd_readvariableop_resource
identity??+sequential_3/dense_4/BiasAdd/ReadVariableOp?*sequential_3/dense_4/MatMul/ReadVariableOp?+sequential_3/dense_5/BiasAdd/ReadVariableOp?*sequential_3/dense_5/MatMul/ReadVariableOp?)sequential_3/embedding_2/embedding_lookup?Mtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2?
 text_vectorization_1/StringLowerStringLowerinputs*'
_output_shapes
:?????????2"
 text_vectorization_1/StringLower?
'text_vectorization_1/StaticRegexReplaceStaticRegexReplace)text_vectorization_1/StringLower:output:0*'
_output_shapes
:?????????*
pattern<br />*
rewrite 2)
'text_vectorization_1/StaticRegexReplace?
)text_vectorization_1/StaticRegexReplace_1StaticRegexReplace0text_vectorization_1/StaticRegexReplace:output:0*'
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite 2+
)text_vectorization_1/StaticRegexReplace_1?
text_vectorization_1/SqueezeSqueeze2text_vectorization_1/StaticRegexReplace_1:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????2
text_vectorization_1/Squeeze?
&text_vectorization_1/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2(
&text_vectorization_1/StringSplit/Const?
.text_vectorization_1/StringSplit/StringSplitV2StringSplitV2%text_vectorization_1/Squeeze:output:0/text_vectorization_1/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:20
.text_vectorization_1/StringSplit/StringSplitV2?
4text_vectorization_1/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        26
4text_vectorization_1/StringSplit/strided_slice/stack?
6text_vectorization_1/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       28
6text_vectorization_1/StringSplit/strided_slice/stack_1?
6text_vectorization_1/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      28
6text_vectorization_1/StringSplit/strided_slice/stack_2?
.text_vectorization_1/StringSplit/strided_sliceStridedSlice8text_vectorization_1/StringSplit/StringSplitV2:indices:0=text_vectorization_1/StringSplit/strided_slice/stack:output:0?text_vectorization_1/StringSplit/strided_slice/stack_1:output:0?text_vectorization_1/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask20
.text_vectorization_1/StringSplit/strided_slice?
6text_vectorization_1/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6text_vectorization_1/StringSplit/strided_slice_1/stack?
8text_vectorization_1/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8text_vectorization_1/StringSplit/strided_slice_1/stack_1?
8text_vectorization_1/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8text_vectorization_1/StringSplit/strided_slice_1/stack_2?
0text_vectorization_1/StringSplit/strided_slice_1StridedSlice6text_vectorization_1/StringSplit/StringSplitV2:shape:0?text_vectorization_1/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_1/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_1/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask22
0text_vectorization_1/StringSplit/strided_slice_1?
Wtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_1/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2Y
Wtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast?
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_1/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2[
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1?
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2c
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape?
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2c
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const?
`text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2b
`text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod?
etext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2g
etext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y?
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2e
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater?
`text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2b
`text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast?
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2e
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1?
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2a
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max?
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2c
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y?
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2a
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add?
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2a
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul?
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2e
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum?
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2e
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum?
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2e
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2?
dtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:?????????2f
dtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount?
^text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2`
^text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis?
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:?????????2[
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum?
btext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2d
btext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0?
^text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2`
^text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis?
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:?????????2[
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat?
Mtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Ztext_vectorization_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handle7text_vectorization_1/StringSplit/StringSplitV2:values:0[text_vectorization_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:?????????2O
Mtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2?
6text_vectorization_1/string_lookup_1/assert_equal/NoOpNoOp*
_output_shapes
 28
6text_vectorization_1/string_lookup_1/assert_equal/NoOp?
-text_vectorization_1/string_lookup_1/IdentityIdentityVtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2/
-text_vectorization_1/string_lookup_1/Identity?
/text_vectorization_1/string_lookup_1/Identity_1Identitybtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:?????????21
/text_vectorization_1/string_lookup_1/Identity_1?
1text_vectorization_1/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 23
1text_vectorization_1/RaggedToTensor/default_value?
)text_vectorization_1/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2+
)text_vectorization_1/RaggedToTensor/Const?
8text_vectorization_1/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_1/RaggedToTensor/Const:output:06text_vectorization_1/string_lookup_1/Identity:output:0:text_vectorization_1/RaggedToTensor/default_value:output:08text_vectorization_1/string_lookup_1/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:??????????????????*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2:
8text_vectorization_1/RaggedToTensor/RaggedTensorToTensor?
text_vectorization_1/ShapeShapeAtext_vectorization_1/RaggedToTensor/RaggedTensorToTensor:result:0*
T0	*
_output_shapes
:2
text_vectorization_1/Shape?
(text_vectorization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(text_vectorization_1/strided_slice/stack?
*text_vectorization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*text_vectorization_1/strided_slice/stack_1?
*text_vectorization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*text_vectorization_1/strided_slice/stack_2?
"text_vectorization_1/strided_sliceStridedSlice#text_vectorization_1/Shape:output:01text_vectorization_1/strided_slice/stack:output:03text_vectorization_1/strided_slice/stack_1:output:03text_vectorization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"text_vectorization_1/strided_slicez
text_vectorization_1/sub/xConst*
_output_shapes
: *
dtype0*
value	B :d2
text_vectorization_1/sub/x?
text_vectorization_1/subSub#text_vectorization_1/sub/x:output:0+text_vectorization_1/strided_slice:output:0*
T0*
_output_shapes
: 2
text_vectorization_1/sub|
text_vectorization_1/Less/yConst*
_output_shapes
: *
dtype0*
value	B :d2
text_vectorization_1/Less/y?
text_vectorization_1/LessLess+text_vectorization_1/strided_slice:output:0$text_vectorization_1/Less/y:output:0*
T0*
_output_shapes
: 2
text_vectorization_1/Less?
text_vectorization_1/condStatelessIftext_vectorization_1/Less:z:0text_vectorization_1/sub:z:0Atext_vectorization_1/RaggedToTensor/RaggedTensorToTensor:result:0*
Tcond0
*
Tin
2	*
Tout
2	*
_lower_using_switch_merge(*0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *9
else_branch*R(
&text_vectorization_1_cond_false_323367*/
output_shapes
:??????????????????*8
then_branch)R'
%text_vectorization_1_cond_true_3233662
text_vectorization_1/cond?
"text_vectorization_1/cond/IdentityIdentity"text_vectorization_1/cond:output:0*
T0	*'
_output_shapes
:?????????d2$
"text_vectorization_1/cond/Identity?
sequential_3/CastCast+text_vectorization_1/cond/Identity:output:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????d2
sequential_3/Cast?
sequential_3/embedding_2/CastCastsequential_3/Cast:y:0*

DstT0*

SrcT0*'
_output_shapes
:?????????d2
sequential_3/embedding_2/Cast?
)sequential_3/embedding_2/embedding_lookupResourceGather0sequential_3_embedding_2_embedding_lookup_323389!sequential_3/embedding_2/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*C
_class9
75loc:@sequential_3/embedding_2/embedding_lookup/323389*+
_output_shapes
:?????????d*
dtype02+
)sequential_3/embedding_2/embedding_lookup?
2sequential_3/embedding_2/embedding_lookup/IdentityIdentity2sequential_3/embedding_2/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@sequential_3/embedding_2/embedding_lookup/323389*+
_output_shapes
:?????????d24
2sequential_3/embedding_2/embedding_lookup/Identity?
4sequential_3/embedding_2/embedding_lookup/Identity_1Identity;sequential_3/embedding_2/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????d26
4sequential_3/embedding_2/embedding_lookup/Identity_1?
sequential_3/dropout_4/IdentityIdentity=sequential_3/embedding_2/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????d2!
sequential_3/dropout_4/Identity?
>sequential_3/global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2@
>sequential_3/global_average_pooling1d_2/Mean/reduction_indices?
,sequential_3/global_average_pooling1d_2/MeanMean(sequential_3/dropout_4/Identity:output:0Gsequential_3/global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2.
,sequential_3/global_average_pooling1d_2/Mean?
sequential_3/dropout_5/IdentityIdentity5sequential_3/global_average_pooling1d_2/Mean:output:0*
T0*'
_output_shapes
:?????????2!
sequential_3/dropout_5/Identity?
*sequential_3/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*sequential_3/dense_4/MatMul/ReadVariableOp?
sequential_3/dense_4/MatMulMatMul(sequential_3/dropout_5/Identity:output:02sequential_3/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_3/dense_4/MatMul?
+sequential_3/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_3/dense_4/BiasAdd/ReadVariableOp?
sequential_3/dense_4/BiasAddBiasAdd%sequential_3/dense_4/MatMul:product:03sequential_3/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_3/dense_4/BiasAdd?
sequential_3/dense_4/ReluRelu%sequential_3/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_3/dense_4/Relu?
*sequential_3/dense_5/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*sequential_3/dense_5/MatMul/ReadVariableOp?
sequential_3/dense_5/MatMulMatMul'sequential_3/dense_4/Relu:activations:02sequential_3/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_3/dense_5/MatMul?
+sequential_3/dense_5/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_3/dense_5/BiasAdd/ReadVariableOp?
sequential_3/dense_5/BiasAddBiasAdd%sequential_3/dense_5/MatMul:product:03sequential_3/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_3/dense_5/BiasAdd?
sequential_3/dense_5/SigmoidSigmoid%sequential_3/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_3/dense_5/Sigmoid?
IdentityIdentity sequential_3/dense_5/Sigmoid:y:0,^sequential_3/dense_4/BiasAdd/ReadVariableOp+^sequential_3/dense_4/MatMul/ReadVariableOp,^sequential_3/dense_5/BiasAdd/ReadVariableOp+^sequential_3/dense_5/MatMul/ReadVariableOp*^sequential_3/embedding_2/embedding_lookupN^text_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:?????????:: :::::2Z
+sequential_3/dense_4/BiasAdd/ReadVariableOp+sequential_3/dense_4/BiasAdd/ReadVariableOp2X
*sequential_3/dense_4/MatMul/ReadVariableOp*sequential_3/dense_4/MatMul/ReadVariableOp2Z
+sequential_3/dense_5/BiasAdd/ReadVariableOp+sequential_3/dense_5/BiasAdd/ReadVariableOp2X
*sequential_3/dense_5/MatMul/ReadVariableOp*sequential_3/dense_5/MatMul/ReadVariableOp2V
)sequential_3/embedding_2/embedding_lookup)sequential_3/embedding_2/embedding_lookup2?
Mtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2Mtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
/
__inference__initializer_323804
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
?
&text_vectorization_1_cond_false_322695)
%text_vectorization_1_cond_placeholderd
`text_vectorization_1_cond_strided_slice_text_vectorization_1_raggedtotensor_raggedtensortotensor	&
"text_vectorization_1_cond_identity	?
-text_vectorization_1/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2/
-text_vectorization_1/cond/strided_slice/stack?
/text_vectorization_1/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   21
/text_vectorization_1/cond/strided_slice/stack_1?
/text_vectorization_1/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/text_vectorization_1/cond/strided_slice/stack_2?
'text_vectorization_1/cond/strided_sliceStridedSlice`text_vectorization_1_cond_strided_slice_text_vectorization_1_raggedtotensor_raggedtensortotensor6text_vectorization_1/cond/strided_slice/stack:output:08text_vectorization_1/cond/strided_slice/stack_1:output:08text_vectorization_1/cond/strided_slice/stack_2:output:0*
Index0*
T0	*0
_output_shapes
:??????????????????*

begin_mask*
end_mask2)
'text_vectorization_1/cond/strided_slice?
"text_vectorization_1/cond/IdentityIdentity0text_vectorization_1/cond/strided_slice:output:0*
T0	*0
_output_shapes
:??????????????????2$
"text_vectorization_1/cond/Identity"Q
"text_vectorization_1_cond_identity+text_vectorization_1/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_322622

inputs
embedding_2_322605
dense_4_322611
dense_4_322613
dense_5_322616
dense_5_322618
identity??dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?#embedding_2/StatefulPartitionedCall?
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_2_322605*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_2_layer_call_and_return_conditional_losses_3224052%
#embedding_2/StatefulPartitionedCall?
dropout_4/PartitionedCallPartitionedCall,embedding_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_3224342
dropout_4/PartitionedCall?
*global_average_pooling1d_2/PartitionedCallPartitionedCall"dropout_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *_
fZRX
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_3224522,
*global_average_pooling1d_2/PartitionedCall?
dropout_5/PartitionedCallPartitionedCall3global_average_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_3224762
dropout_5/PartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_4_322611dense_4_322613*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_3225002!
dense_4/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_322616dense_5_322618*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_3225272!
dense_5/StatefulPartitionedCall?
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????:::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
-__inference_sequential_3_layer_call_fn_323661

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_3226222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_322587

inputs
embedding_2_322570
dense_4_322576
dense_4_322578
dense_5_322581
dense_5_322583
identity??dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?!dropout_4/StatefulPartitionedCall?!dropout_5/StatefulPartitionedCall?#embedding_2/StatefulPartitionedCall?
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_2_322570*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_2_layer_call_and_return_conditional_losses_3224052%
#embedding_2/StatefulPartitionedCall?
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall,embedding_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_3224292#
!dropout_4/StatefulPartitionedCall?
*global_average_pooling1d_2/PartitionedCallPartitionedCall*dropout_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *_
fZRX
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_3224522,
*global_average_pooling1d_2/PartitionedCall?
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_2/PartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_3224712#
!dropout_5/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_4_322576dense_4_322578*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_3225002!
dense_4/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_322581dense_5_322583*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_3225272!
dense_5/StatefulPartitionedCall?
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????:::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
W
;__inference_global_average_pooling1d_2_layer_call_fn_323716

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *_
fZRX
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_3224522
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?D
?
__inference__traced_save_323955
file_prefix5
1savev2_embedding_2_embeddings_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableopU
Qsavev2_string_lookup_1_index_table_lookup_table_export_values_lookuptableexportv2W
Ssavev2_string_lookup_1_index_table_lookup_table_export_values_lookuptableexportv2_1	(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableop<
8savev2_adam_embedding_2_embeddings_m_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableop4
0savev2_adam_dense_5_kernel_m_read_readvariableop2
.savev2_adam_dense_5_bias_m_read_readvariableop<
8savev2_adam_embedding_2_embeddings_v_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableop4
0savev2_adam_dense_5_kernel_v_read_readvariableop2
.savev2_adam_dense_5_bias_v_read_readvariableop
savev2_const_1

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_index_lookup_layer/_table/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_index_lookup_layer/_table/.ATTRIBUTES/table-valuesB>layer_with_weights-1/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-1/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/0/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/1/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/2/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/3/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/4/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/0/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/1/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/2/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/3/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/4/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_embedding_2_embeddings_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableopQsavev2_string_lookup_1_index_table_lookup_table_export_values_lookuptableexportv2Ssavev2_string_lookup_1_index_table_lookup_table_export_values_lookuptableexportv2_1$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop8savev2_adam_embedding_2_embeddings_m_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop0savev2_adam_dense_5_kernel_m_read_readvariableop.savev2_adam_dense_5_bias_m_read_readvariableop8savev2_adam_embedding_2_embeddings_v_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableop0savev2_adam_dense_5_kernel_v_read_readvariableop.savev2_adam_dense_5_bias_v_read_readvariableopsavev2_const_1"/device:CPU:0*
_output_shapes
 *-
dtypes#
!2		2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??::::::: : : : : : : : : : : : : :
??:::::
??::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::&"
 
_output_shapes
:
??:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
?
L
__inference__creator_323799
identity??string_lookup_1_index_table?
string_lookup_1_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_226780*
value_dtype0	2
string_lookup_1_index_table?
IdentityIdentity*string_lookup_1_index_table:table_handle:0^string_lookup_1_index_table*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 2:
string_lookup_1_index_tablestring_lookup_1_index_table
?
?
%text_vectorization_1_cond_true_323091E
Atext_vectorization_1_cond_pad_paddings_1_text_vectorization_1_subZ
Vtext_vectorization_1_cond_pad_text_vectorization_1_raggedtotensor_raggedtensortotensor	&
"text_vectorization_1_cond_identity	?
*text_vectorization_1/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2,
*text_vectorization_1/cond/Pad/paddings/1/0?
(text_vectorization_1/cond/Pad/paddings/1Pack3text_vectorization_1/cond/Pad/paddings/1/0:output:0Atext_vectorization_1_cond_pad_paddings_1_text_vectorization_1_sub*
N*
T0*
_output_shapes
:2*
(text_vectorization_1/cond/Pad/paddings/1?
*text_vectorization_1/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*text_vectorization_1/cond/Pad/paddings/0_1?
&text_vectorization_1/cond/Pad/paddingsPack3text_vectorization_1/cond/Pad/paddings/0_1:output:01text_vectorization_1/cond/Pad/paddings/1:output:0*
N*
T0*
_output_shapes

:2(
&text_vectorization_1/cond/Pad/paddings?
text_vectorization_1/cond/PadPadVtext_vectorization_1_cond_pad_text_vectorization_1_raggedtotensor_raggedtensortotensor/text_vectorization_1/cond/Pad/paddings:output:0*
T0	*0
_output_shapes
:??????????????????2
text_vectorization_1/cond/Pad?
"text_vectorization_1/cond/IdentityIdentity&text_vectorization_1/cond/Pad:output:0*
T0	*0
_output_shapes
:??????????????????2$
"text_vectorization_1/cond/Identity"Q
"text_vectorization_1_cond_identity+text_vectorization_1/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
??
?
H__inference_sequential_4_layer_call_and_return_conditional_losses_322920
text_vectorization_1_input^
Ztext_vectorization_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handle_
[text_vectorization_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value	
sequential_3_322908
sequential_3_322910
sequential_3_322912
sequential_3_322914
sequential_3_322916
identity??$sequential_3/StatefulPartitionedCall?Mtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2?
 text_vectorization_1/StringLowerStringLowertext_vectorization_1_input*'
_output_shapes
:?????????2"
 text_vectorization_1/StringLower?
'text_vectorization_1/StaticRegexReplaceStaticRegexReplace)text_vectorization_1/StringLower:output:0*'
_output_shapes
:?????????*
pattern<br />*
rewrite 2)
'text_vectorization_1/StaticRegexReplace?
)text_vectorization_1/StaticRegexReplace_1StaticRegexReplace0text_vectorization_1/StaticRegexReplace:output:0*'
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite 2+
)text_vectorization_1/StaticRegexReplace_1?
text_vectorization_1/SqueezeSqueeze2text_vectorization_1/StaticRegexReplace_1:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????2
text_vectorization_1/Squeeze?
&text_vectorization_1/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2(
&text_vectorization_1/StringSplit/Const?
.text_vectorization_1/StringSplit/StringSplitV2StringSplitV2%text_vectorization_1/Squeeze:output:0/text_vectorization_1/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:20
.text_vectorization_1/StringSplit/StringSplitV2?
4text_vectorization_1/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        26
4text_vectorization_1/StringSplit/strided_slice/stack?
6text_vectorization_1/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       28
6text_vectorization_1/StringSplit/strided_slice/stack_1?
6text_vectorization_1/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      28
6text_vectorization_1/StringSplit/strided_slice/stack_2?
.text_vectorization_1/StringSplit/strided_sliceStridedSlice8text_vectorization_1/StringSplit/StringSplitV2:indices:0=text_vectorization_1/StringSplit/strided_slice/stack:output:0?text_vectorization_1/StringSplit/strided_slice/stack_1:output:0?text_vectorization_1/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask20
.text_vectorization_1/StringSplit/strided_slice?
6text_vectorization_1/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6text_vectorization_1/StringSplit/strided_slice_1/stack?
8text_vectorization_1/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8text_vectorization_1/StringSplit/strided_slice_1/stack_1?
8text_vectorization_1/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8text_vectorization_1/StringSplit/strided_slice_1/stack_2?
0text_vectorization_1/StringSplit/strided_slice_1StridedSlice6text_vectorization_1/StringSplit/StringSplitV2:shape:0?text_vectorization_1/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_1/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_1/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask22
0text_vectorization_1/StringSplit/strided_slice_1?
Wtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_1/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2Y
Wtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast?
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_1/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2[
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1?
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2c
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape?
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2c
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const?
`text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2b
`text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod?
etext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2g
etext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y?
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2e
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater?
`text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2b
`text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast?
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2e
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1?
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2a
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max?
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2c
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y?
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2a
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add?
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2a
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul?
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2e
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum?
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2e
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum?
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2e
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2?
dtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:?????????2f
dtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount?
^text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2`
^text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis?
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:?????????2[
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum?
btext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2d
btext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0?
^text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2`
^text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis?
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:?????????2[
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat?
Mtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Ztext_vectorization_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handle7text_vectorization_1/StringSplit/StringSplitV2:values:0[text_vectorization_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:?????????2O
Mtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2?
6text_vectorization_1/string_lookup_1/assert_equal/NoOpNoOp*
_output_shapes
 28
6text_vectorization_1/string_lookup_1/assert_equal/NoOp?
-text_vectorization_1/string_lookup_1/IdentityIdentityVtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2/
-text_vectorization_1/string_lookup_1/Identity?
/text_vectorization_1/string_lookup_1/Identity_1Identitybtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:?????????21
/text_vectorization_1/string_lookup_1/Identity_1?
1text_vectorization_1/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 23
1text_vectorization_1/RaggedToTensor/default_value?
)text_vectorization_1/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2+
)text_vectorization_1/RaggedToTensor/Const?
8text_vectorization_1/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_1/RaggedToTensor/Const:output:06text_vectorization_1/string_lookup_1/Identity:output:0:text_vectorization_1/RaggedToTensor/default_value:output:08text_vectorization_1/string_lookup_1/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:??????????????????*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2:
8text_vectorization_1/RaggedToTensor/RaggedTensorToTensor?
text_vectorization_1/ShapeShapeAtext_vectorization_1/RaggedToTensor/RaggedTensorToTensor:result:0*
T0	*
_output_shapes
:2
text_vectorization_1/Shape?
(text_vectorization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(text_vectorization_1/strided_slice/stack?
*text_vectorization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*text_vectorization_1/strided_slice/stack_1?
*text_vectorization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*text_vectorization_1/strided_slice/stack_2?
"text_vectorization_1/strided_sliceStridedSlice#text_vectorization_1/Shape:output:01text_vectorization_1/strided_slice/stack:output:03text_vectorization_1/strided_slice/stack_1:output:03text_vectorization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"text_vectorization_1/strided_slicez
text_vectorization_1/sub/xConst*
_output_shapes
: *
dtype0*
value	B :d2
text_vectorization_1/sub/x?
text_vectorization_1/subSub#text_vectorization_1/sub/x:output:0+text_vectorization_1/strided_slice:output:0*
T0*
_output_shapes
: 2
text_vectorization_1/sub|
text_vectorization_1/Less/yConst*
_output_shapes
: *
dtype0*
value	B :d2
text_vectorization_1/Less/y?
text_vectorization_1/LessLess+text_vectorization_1/strided_slice:output:0$text_vectorization_1/Less/y:output:0*
T0*
_output_shapes
: 2
text_vectorization_1/Less?
text_vectorization_1/condStatelessIftext_vectorization_1/Less:z:0text_vectorization_1/sub:z:0Atext_vectorization_1/RaggedToTensor/RaggedTensorToTensor:result:0*
Tcond0
*
Tin
2	*
Tout
2	*
_lower_using_switch_merge(*0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *9
else_branch*R(
&text_vectorization_1_cond_false_322888*/
output_shapes
:??????????????????*8
then_branch)R'
%text_vectorization_1_cond_true_3228872
text_vectorization_1/cond?
"text_vectorization_1/cond/IdentityIdentity"text_vectorization_1/cond:output:0*
T0	*'
_output_shapes
:?????????d2$
"text_vectorization_1/cond/Identity?
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall+text_vectorization_1/cond/Identity:output:0sequential_3_322908sequential_3_322910sequential_3_322912sequential_3_322914sequential_3_322916*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_3227862&
$sequential_3/StatefulPartitionedCall?
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:0%^sequential_3/StatefulPartitionedCallN^text_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:?????????:: :::::2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall2?
Mtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2Mtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2:c _
'
_output_shapes
:?????????
4
_user_specified_nametext_vectorization_1_input:

_output_shapes
: 
?
?
%text_vectorization_1_cond_true_322981E
Atext_vectorization_1_cond_pad_paddings_1_text_vectorization_1_subZ
Vtext_vectorization_1_cond_pad_text_vectorization_1_raggedtotensor_raggedtensortotensor	&
"text_vectorization_1_cond_identity	?
*text_vectorization_1/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2,
*text_vectorization_1/cond/Pad/paddings/1/0?
(text_vectorization_1/cond/Pad/paddings/1Pack3text_vectorization_1/cond/Pad/paddings/1/0:output:0Atext_vectorization_1_cond_pad_paddings_1_text_vectorization_1_sub*
N*
T0*
_output_shapes
:2*
(text_vectorization_1/cond/Pad/paddings/1?
*text_vectorization_1/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*text_vectorization_1/cond/Pad/paddings/0_1?
&text_vectorization_1/cond/Pad/paddingsPack3text_vectorization_1/cond/Pad/paddings/0_1:output:01text_vectorization_1/cond/Pad/paddings/1:output:0*
N*
T0*
_output_shapes

:2(
&text_vectorization_1/cond/Pad/paddings?
text_vectorization_1/cond/PadPadVtext_vectorization_1_cond_pad_text_vectorization_1_raggedtotensor_raggedtensortotensor/text_vectorization_1/cond/Pad/paddings:output:0*
T0	*0
_output_shapes
:??????????????????2
text_vectorization_1/cond/Pad?
"text_vectorization_1/cond/IdentityIdentity&text_vectorization_1/cond/Pad:output:0*
T0	*0
_output_shapes
:??????????????????2$
"text_vectorization_1/cond/Identity"Q
"text_vectorization_1_cond_identity+text_vectorization_1/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?#
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_323631

inputs'
#embedding_2_embedding_lookup_323607*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identity??dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?embedding_2/embedding_lookup~
embedding_2/CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:??????????????????2
embedding_2/Cast?
embedding_2/embedding_lookupResourceGather#embedding_2_embedding_lookup_323607embedding_2/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_2/embedding_lookup/323607*4
_output_shapes"
 :??????????????????*
dtype02
embedding_2/embedding_lookup?
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_2/embedding_lookup/323607*4
_output_shapes"
 :??????????????????2'
%embedding_2/embedding_lookup/Identity?
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :??????????????????2)
'embedding_2/embedding_lookup/Identity_1?
dropout_4/IdentityIdentity0embedding_2/embedding_lookup/Identity_1:output:0*
T0*4
_output_shapes"
 :??????????????????2
dropout_4/Identity?
1global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_2/Mean/reduction_indices?
global_average_pooling1d_2/MeanMeandropout_4/Identity:output:0:global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2!
global_average_pooling1d_2/Mean?
dropout_5/IdentityIdentity(global_average_pooling1d_2/Mean:output:0*
T0*'
_output_shapes
:?????????2
dropout_5/Identity?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMuldropout_5/Identity:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_4/Relu?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/BiasAddy
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_5/Sigmoid?
IdentityIdentitydense_5/Sigmoid:y:0^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^embedding_2/embedding_lookup*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????:::::2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2<
embedding_2/embedding_lookupembedding_2/embedding_lookup:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?6
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_323502

inputs	'
#embedding_2_embedding_lookup_323464*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identity??dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?embedding_2/embedding_lookup]
CastCastinputs*

DstT0*

SrcT0	*'
_output_shapes
:?????????d2
Castw
embedding_2/CastCastCast:y:0*

DstT0*

SrcT0*'
_output_shapes
:?????????d2
embedding_2/Cast?
embedding_2/embedding_lookupResourceGather#embedding_2_embedding_lookup_323464embedding_2/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_2/embedding_lookup/323464*+
_output_shapes
:?????????d*
dtype02
embedding_2/embedding_lookup?
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_2/embedding_lookup/323464*+
_output_shapes
:?????????d2'
%embedding_2/embedding_lookup/Identity?
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????d2)
'embedding_2/embedding_lookup/Identity_1w
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_4/dropout/Const?
dropout_4/dropout/MulMul0embedding_2/embedding_lookup/Identity_1:output:0 dropout_4/dropout/Const:output:0*
T0*+
_output_shapes
:?????????d2
dropout_4/dropout/Mul?
dropout_4/dropout/ShapeShape0embedding_2/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
dropout_4/dropout/Shape?
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????d*
dtype020
.dropout_4/dropout/random_uniform/RandomUniform?
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 dropout_4/dropout/GreaterEqual/y?
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????d2 
dropout_4/dropout/GreaterEqual?
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????d2
dropout_4/dropout/Cast?
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????d2
dropout_4/dropout/Mul_1?
1global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_2/Mean/reduction_indices?
global_average_pooling1d_2/MeanMeandropout_4/dropout/Mul_1:z:0:global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2!
global_average_pooling1d_2/Meanw
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_5/dropout/Const?
dropout_5/dropout/MulMul(global_average_pooling1d_2/Mean:output:0 dropout_5/dropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout_5/dropout/Mul?
dropout_5/dropout/ShapeShape(global_average_pooling1d_2/Mean:output:0*
T0*
_output_shapes
:2
dropout_5/dropout/Shape?
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype020
.dropout_5/dropout/random_uniform/RandomUniform?
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 dropout_5/dropout/GreaterEqual/y?
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2 
dropout_5/dropout/GreaterEqual?
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout_5/dropout/Cast?
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout_5/dropout/Mul_1?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMuldropout_5/dropout/Mul_1:z:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_4/Relu?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/BiasAddy
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_5/Sigmoid?
IdentityIdentitydense_5/Sigmoid:y:0^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^embedding_2/embedding_lookup*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????d:::::2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2<
embedding_2/embedding_lookupembedding_2/embedding_lookup:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
+
__inference_<lambda>_323841
identityS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
?
%text_vectorization_1_cond_true_322887E
Atext_vectorization_1_cond_pad_paddings_1_text_vectorization_1_subZ
Vtext_vectorization_1_cond_pad_text_vectorization_1_raggedtotensor_raggedtensortotensor	&
"text_vectorization_1_cond_identity	?
*text_vectorization_1/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2,
*text_vectorization_1/cond/Pad/paddings/1/0?
(text_vectorization_1/cond/Pad/paddings/1Pack3text_vectorization_1/cond/Pad/paddings/1/0:output:0Atext_vectorization_1_cond_pad_paddings_1_text_vectorization_1_sub*
N*
T0*
_output_shapes
:2*
(text_vectorization_1/cond/Pad/paddings/1?
*text_vectorization_1/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*text_vectorization_1/cond/Pad/paddings/0_1?
&text_vectorization_1/cond/Pad/paddingsPack3text_vectorization_1/cond/Pad/paddings/0_1:output:01text_vectorization_1/cond/Pad/paddings/1:output:0*
N*
T0*
_output_shapes

:2(
&text_vectorization_1/cond/Pad/paddings?
text_vectorization_1/cond/PadPadVtext_vectorization_1_cond_pad_text_vectorization_1_raggedtotensor_raggedtensortotensor/text_vectorization_1/cond/Pad/paddings:output:0*
T0	*0
_output_shapes
:??????????????????2
text_vectorization_1/cond/Pad?
"text_vectorization_1/cond/IdentityIdentity&text_vectorization_1/cond/Pad:output:0*
T0	*0
_output_shapes
:??????????????????2$
"text_vectorization_1/cond/Identity"Q
"text_vectorization_1_cond_identity+text_vectorization_1/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_322564
embedding_2_input
embedding_2_322547
dense_4_322553
dense_4_322555
dense_5_322558
dense_5_322560
identity??dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?#embedding_2/StatefulPartitionedCall?
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallembedding_2_inputembedding_2_322547*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_2_layer_call_and_return_conditional_losses_3224052%
#embedding_2/StatefulPartitionedCall?
dropout_4/PartitionedCallPartitionedCall,embedding_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_3224342
dropout_4/PartitionedCall?
*global_average_pooling1d_2/PartitionedCallPartitionedCall"dropout_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *_
fZRX
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_3224522,
*global_average_pooling1d_2/PartitionedCall?
dropout_5/PartitionedCallPartitionedCall3global_average_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_3224762
dropout_5/PartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_4_322553dense_4_322555*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_3225002!
dense_4/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_322558dense_5_322560*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_3225272!
dense_5/StatefulPartitionedCall?
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????:::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall:c _
0
_output_shapes
:??????????????????
+
_user_specified_nameembedding_2_input
?
c
*__inference_dropout_5_layer_call_fn_323749

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_3224712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_5_layer_call_and_return_conditional_losses_323739

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
H__inference_sequential_4_layer_call_and_return_conditional_losses_323124

inputs^
Ztext_vectorization_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handle_
[text_vectorization_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value	
sequential_3_323112
sequential_3_323114
sequential_3_323116
sequential_3_323118
sequential_3_323120
identity??$sequential_3/StatefulPartitionedCall?Mtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2?
 text_vectorization_1/StringLowerStringLowerinputs*'
_output_shapes
:?????????2"
 text_vectorization_1/StringLower?
'text_vectorization_1/StaticRegexReplaceStaticRegexReplace)text_vectorization_1/StringLower:output:0*'
_output_shapes
:?????????*
pattern<br />*
rewrite 2)
'text_vectorization_1/StaticRegexReplace?
)text_vectorization_1/StaticRegexReplace_1StaticRegexReplace0text_vectorization_1/StaticRegexReplace:output:0*'
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite 2+
)text_vectorization_1/StaticRegexReplace_1?
text_vectorization_1/SqueezeSqueeze2text_vectorization_1/StaticRegexReplace_1:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????2
text_vectorization_1/Squeeze?
&text_vectorization_1/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2(
&text_vectorization_1/StringSplit/Const?
.text_vectorization_1/StringSplit/StringSplitV2StringSplitV2%text_vectorization_1/Squeeze:output:0/text_vectorization_1/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:20
.text_vectorization_1/StringSplit/StringSplitV2?
4text_vectorization_1/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        26
4text_vectorization_1/StringSplit/strided_slice/stack?
6text_vectorization_1/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       28
6text_vectorization_1/StringSplit/strided_slice/stack_1?
6text_vectorization_1/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      28
6text_vectorization_1/StringSplit/strided_slice/stack_2?
.text_vectorization_1/StringSplit/strided_sliceStridedSlice8text_vectorization_1/StringSplit/StringSplitV2:indices:0=text_vectorization_1/StringSplit/strided_slice/stack:output:0?text_vectorization_1/StringSplit/strided_slice/stack_1:output:0?text_vectorization_1/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask20
.text_vectorization_1/StringSplit/strided_slice?
6text_vectorization_1/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6text_vectorization_1/StringSplit/strided_slice_1/stack?
8text_vectorization_1/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8text_vectorization_1/StringSplit/strided_slice_1/stack_1?
8text_vectorization_1/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8text_vectorization_1/StringSplit/strided_slice_1/stack_2?
0text_vectorization_1/StringSplit/strided_slice_1StridedSlice6text_vectorization_1/StringSplit/StringSplitV2:shape:0?text_vectorization_1/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_1/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_1/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask22
0text_vectorization_1/StringSplit/strided_slice_1?
Wtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_1/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2Y
Wtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast?
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_1/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2[
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1?
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2c
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape?
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2c
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const?
`text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2b
`text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod?
etext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2g
etext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y?
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2e
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater?
`text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2b
`text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast?
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2e
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1?
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2a
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max?
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2c
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y?
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2a
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add?
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2a
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul?
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2e
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum?
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2e
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum?
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2e
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2?
dtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:?????????2f
dtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount?
^text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2`
^text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis?
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:?????????2[
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum?
btext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2d
btext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0?
^text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2`
^text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis?
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:?????????2[
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat?
Mtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Ztext_vectorization_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handle7text_vectorization_1/StringSplit/StringSplitV2:values:0[text_vectorization_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:?????????2O
Mtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2?
6text_vectorization_1/string_lookup_1/assert_equal/NoOpNoOp*
_output_shapes
 28
6text_vectorization_1/string_lookup_1/assert_equal/NoOp?
-text_vectorization_1/string_lookup_1/IdentityIdentityVtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2/
-text_vectorization_1/string_lookup_1/Identity?
/text_vectorization_1/string_lookup_1/Identity_1Identitybtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:?????????21
/text_vectorization_1/string_lookup_1/Identity_1?
1text_vectorization_1/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 23
1text_vectorization_1/RaggedToTensor/default_value?
)text_vectorization_1/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2+
)text_vectorization_1/RaggedToTensor/Const?
8text_vectorization_1/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_1/RaggedToTensor/Const:output:06text_vectorization_1/string_lookup_1/Identity:output:0:text_vectorization_1/RaggedToTensor/default_value:output:08text_vectorization_1/string_lookup_1/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:??????????????????*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2:
8text_vectorization_1/RaggedToTensor/RaggedTensorToTensor?
text_vectorization_1/ShapeShapeAtext_vectorization_1/RaggedToTensor/RaggedTensorToTensor:result:0*
T0	*
_output_shapes
:2
text_vectorization_1/Shape?
(text_vectorization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(text_vectorization_1/strided_slice/stack?
*text_vectorization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*text_vectorization_1/strided_slice/stack_1?
*text_vectorization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*text_vectorization_1/strided_slice/stack_2?
"text_vectorization_1/strided_sliceStridedSlice#text_vectorization_1/Shape:output:01text_vectorization_1/strided_slice/stack:output:03text_vectorization_1/strided_slice/stack_1:output:03text_vectorization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"text_vectorization_1/strided_slicez
text_vectorization_1/sub/xConst*
_output_shapes
: *
dtype0*
value	B :d2
text_vectorization_1/sub/x?
text_vectorization_1/subSub#text_vectorization_1/sub/x:output:0+text_vectorization_1/strided_slice:output:0*
T0*
_output_shapes
: 2
text_vectorization_1/sub|
text_vectorization_1/Less/yConst*
_output_shapes
: *
dtype0*
value	B :d2
text_vectorization_1/Less/y?
text_vectorization_1/LessLess+text_vectorization_1/strided_slice:output:0$text_vectorization_1/Less/y:output:0*
T0*
_output_shapes
: 2
text_vectorization_1/Less?
text_vectorization_1/condStatelessIftext_vectorization_1/Less:z:0text_vectorization_1/sub:z:0Atext_vectorization_1/RaggedToTensor/RaggedTensorToTensor:result:0*
Tcond0
*
Tin
2	*
Tout
2	*
_lower_using_switch_merge(*0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *9
else_branch*R(
&text_vectorization_1_cond_false_323092*/
output_shapes
:??????????????????*8
then_branch)R'
%text_vectorization_1_cond_true_3230912
text_vectorization_1/cond?
"text_vectorization_1/cond/IdentityIdentity"text_vectorization_1/cond:output:0*
T0	*'
_output_shapes
:?????????d2$
"text_vectorization_1/cond/Identity?
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall+text_vectorization_1/cond/Identity:output:0sequential_3_323112sequential_3_323114sequential_3_323116sequential_3_323118sequential_3_323120*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_3227862&
$sequential_3/StatefulPartitionedCall?
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:0%^sequential_3/StatefulPartitionedCallN^text_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:?????????:: :::::2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall2?
Mtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2Mtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?	
?
C__inference_dense_4_layer_call_and_return_conditional_losses_323765

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_2_layer_call_and_return_conditional_losses_322405

inputs
embedding_lookup_322399
identity??embedding_lookupf
CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:??????????????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_322399Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0**
_class 
loc:@embedding_lookup/322399*4
_output_shapes"
 :??????????????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@embedding_lookup/322399*4
_output_shapes"
 :??????????????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :??????????????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:2$
embedding_lookupembedding_lookup:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
r
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_323722

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
??
?
H__inference_sequential_4_layer_call_and_return_conditional_losses_323308

inputs^
Ztext_vectorization_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handle_
[text_vectorization_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value	4
0sequential_3_embedding_2_embedding_lookup_3232707
3sequential_3_dense_4_matmul_readvariableop_resource8
4sequential_3_dense_4_biasadd_readvariableop_resource7
3sequential_3_dense_5_matmul_readvariableop_resource8
4sequential_3_dense_5_biasadd_readvariableop_resource
identity??+sequential_3/dense_4/BiasAdd/ReadVariableOp?*sequential_3/dense_4/MatMul/ReadVariableOp?+sequential_3/dense_5/BiasAdd/ReadVariableOp?*sequential_3/dense_5/MatMul/ReadVariableOp?)sequential_3/embedding_2/embedding_lookup?Mtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2?
 text_vectorization_1/StringLowerStringLowerinputs*'
_output_shapes
:?????????2"
 text_vectorization_1/StringLower?
'text_vectorization_1/StaticRegexReplaceStaticRegexReplace)text_vectorization_1/StringLower:output:0*'
_output_shapes
:?????????*
pattern<br />*
rewrite 2)
'text_vectorization_1/StaticRegexReplace?
)text_vectorization_1/StaticRegexReplace_1StaticRegexReplace0text_vectorization_1/StaticRegexReplace:output:0*'
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite 2+
)text_vectorization_1/StaticRegexReplace_1?
text_vectorization_1/SqueezeSqueeze2text_vectorization_1/StaticRegexReplace_1:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????2
text_vectorization_1/Squeeze?
&text_vectorization_1/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2(
&text_vectorization_1/StringSplit/Const?
.text_vectorization_1/StringSplit/StringSplitV2StringSplitV2%text_vectorization_1/Squeeze:output:0/text_vectorization_1/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:20
.text_vectorization_1/StringSplit/StringSplitV2?
4text_vectorization_1/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        26
4text_vectorization_1/StringSplit/strided_slice/stack?
6text_vectorization_1/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       28
6text_vectorization_1/StringSplit/strided_slice/stack_1?
6text_vectorization_1/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      28
6text_vectorization_1/StringSplit/strided_slice/stack_2?
.text_vectorization_1/StringSplit/strided_sliceStridedSlice8text_vectorization_1/StringSplit/StringSplitV2:indices:0=text_vectorization_1/StringSplit/strided_slice/stack:output:0?text_vectorization_1/StringSplit/strided_slice/stack_1:output:0?text_vectorization_1/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask20
.text_vectorization_1/StringSplit/strided_slice?
6text_vectorization_1/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6text_vectorization_1/StringSplit/strided_slice_1/stack?
8text_vectorization_1/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8text_vectorization_1/StringSplit/strided_slice_1/stack_1?
8text_vectorization_1/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8text_vectorization_1/StringSplit/strided_slice_1/stack_2?
0text_vectorization_1/StringSplit/strided_slice_1StridedSlice6text_vectorization_1/StringSplit/StringSplitV2:shape:0?text_vectorization_1/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_1/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_1/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask22
0text_vectorization_1/StringSplit/strided_slice_1?
Wtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_1/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2Y
Wtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast?
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_1/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2[
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1?
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2c
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape?
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2c
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const?
`text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2b
`text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod?
etext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2g
etext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y?
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2e
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater?
`text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2b
`text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast?
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2e
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1?
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2a
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max?
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2c
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y?
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2a
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add?
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2a
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul?
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2e
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum?
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2e
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum?
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2e
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2?
dtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:?????????2f
dtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount?
^text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2`
^text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis?
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:?????????2[
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum?
btext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2d
btext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0?
^text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2`
^text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis?
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:?????????2[
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat?
Mtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Ztext_vectorization_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handle7text_vectorization_1/StringSplit/StringSplitV2:values:0[text_vectorization_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:?????????2O
Mtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2?
6text_vectorization_1/string_lookup_1/assert_equal/NoOpNoOp*
_output_shapes
 28
6text_vectorization_1/string_lookup_1/assert_equal/NoOp?
-text_vectorization_1/string_lookup_1/IdentityIdentityVtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2/
-text_vectorization_1/string_lookup_1/Identity?
/text_vectorization_1/string_lookup_1/Identity_1Identitybtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:?????????21
/text_vectorization_1/string_lookup_1/Identity_1?
1text_vectorization_1/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 23
1text_vectorization_1/RaggedToTensor/default_value?
)text_vectorization_1/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2+
)text_vectorization_1/RaggedToTensor/Const?
8text_vectorization_1/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_1/RaggedToTensor/Const:output:06text_vectorization_1/string_lookup_1/Identity:output:0:text_vectorization_1/RaggedToTensor/default_value:output:08text_vectorization_1/string_lookup_1/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:??????????????????*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2:
8text_vectorization_1/RaggedToTensor/RaggedTensorToTensor?
text_vectorization_1/ShapeShapeAtext_vectorization_1/RaggedToTensor/RaggedTensorToTensor:result:0*
T0	*
_output_shapes
:2
text_vectorization_1/Shape?
(text_vectorization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(text_vectorization_1/strided_slice/stack?
*text_vectorization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*text_vectorization_1/strided_slice/stack_1?
*text_vectorization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*text_vectorization_1/strided_slice/stack_2?
"text_vectorization_1/strided_sliceStridedSlice#text_vectorization_1/Shape:output:01text_vectorization_1/strided_slice/stack:output:03text_vectorization_1/strided_slice/stack_1:output:03text_vectorization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"text_vectorization_1/strided_slicez
text_vectorization_1/sub/xConst*
_output_shapes
: *
dtype0*
value	B :d2
text_vectorization_1/sub/x?
text_vectorization_1/subSub#text_vectorization_1/sub/x:output:0+text_vectorization_1/strided_slice:output:0*
T0*
_output_shapes
: 2
text_vectorization_1/sub|
text_vectorization_1/Less/yConst*
_output_shapes
: *
dtype0*
value	B :d2
text_vectorization_1/Less/y?
text_vectorization_1/LessLess+text_vectorization_1/strided_slice:output:0$text_vectorization_1/Less/y:output:0*
T0*
_output_shapes
: 2
text_vectorization_1/Less?
text_vectorization_1/condStatelessIftext_vectorization_1/Less:z:0text_vectorization_1/sub:z:0Atext_vectorization_1/RaggedToTensor/RaggedTensorToTensor:result:0*
Tcond0
*
Tin
2	*
Tout
2	*
_lower_using_switch_merge(*0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *9
else_branch*R(
&text_vectorization_1_cond_false_323248*/
output_shapes
:??????????????????*8
then_branch)R'
%text_vectorization_1_cond_true_3232472
text_vectorization_1/cond?
"text_vectorization_1/cond/IdentityIdentity"text_vectorization_1/cond:output:0*
T0	*'
_output_shapes
:?????????d2$
"text_vectorization_1/cond/Identity?
sequential_3/CastCast+text_vectorization_1/cond/Identity:output:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????d2
sequential_3/Cast?
sequential_3/embedding_2/CastCastsequential_3/Cast:y:0*

DstT0*

SrcT0*'
_output_shapes
:?????????d2
sequential_3/embedding_2/Cast?
)sequential_3/embedding_2/embedding_lookupResourceGather0sequential_3_embedding_2_embedding_lookup_323270!sequential_3/embedding_2/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*C
_class9
75loc:@sequential_3/embedding_2/embedding_lookup/323270*+
_output_shapes
:?????????d*
dtype02+
)sequential_3/embedding_2/embedding_lookup?
2sequential_3/embedding_2/embedding_lookup/IdentityIdentity2sequential_3/embedding_2/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@sequential_3/embedding_2/embedding_lookup/323270*+
_output_shapes
:?????????d24
2sequential_3/embedding_2/embedding_lookup/Identity?
4sequential_3/embedding_2/embedding_lookup/Identity_1Identity;sequential_3/embedding_2/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????d26
4sequential_3/embedding_2/embedding_lookup/Identity_1?
$sequential_3/dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$sequential_3/dropout_4/dropout/Const?
"sequential_3/dropout_4/dropout/MulMul=sequential_3/embedding_2/embedding_lookup/Identity_1:output:0-sequential_3/dropout_4/dropout/Const:output:0*
T0*+
_output_shapes
:?????????d2$
"sequential_3/dropout_4/dropout/Mul?
$sequential_3/dropout_4/dropout/ShapeShape=sequential_3/embedding_2/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2&
$sequential_3/dropout_4/dropout/Shape?
;sequential_3/dropout_4/dropout/random_uniform/RandomUniformRandomUniform-sequential_3/dropout_4/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????d*
dtype02=
;sequential_3/dropout_4/dropout/random_uniform/RandomUniform?
-sequential_3/dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2/
-sequential_3/dropout_4/dropout/GreaterEqual/y?
+sequential_3/dropout_4/dropout/GreaterEqualGreaterEqualDsequential_3/dropout_4/dropout/random_uniform/RandomUniform:output:06sequential_3/dropout_4/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????d2-
+sequential_3/dropout_4/dropout/GreaterEqual?
#sequential_3/dropout_4/dropout/CastCast/sequential_3/dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????d2%
#sequential_3/dropout_4/dropout/Cast?
$sequential_3/dropout_4/dropout/Mul_1Mul&sequential_3/dropout_4/dropout/Mul:z:0'sequential_3/dropout_4/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????d2&
$sequential_3/dropout_4/dropout/Mul_1?
>sequential_3/global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2@
>sequential_3/global_average_pooling1d_2/Mean/reduction_indices?
,sequential_3/global_average_pooling1d_2/MeanMean(sequential_3/dropout_4/dropout/Mul_1:z:0Gsequential_3/global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2.
,sequential_3/global_average_pooling1d_2/Mean?
$sequential_3/dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$sequential_3/dropout_5/dropout/Const?
"sequential_3/dropout_5/dropout/MulMul5sequential_3/global_average_pooling1d_2/Mean:output:0-sequential_3/dropout_5/dropout/Const:output:0*
T0*'
_output_shapes
:?????????2$
"sequential_3/dropout_5/dropout/Mul?
$sequential_3/dropout_5/dropout/ShapeShape5sequential_3/global_average_pooling1d_2/Mean:output:0*
T0*
_output_shapes
:2&
$sequential_3/dropout_5/dropout/Shape?
;sequential_3/dropout_5/dropout/random_uniform/RandomUniformRandomUniform-sequential_3/dropout_5/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02=
;sequential_3/dropout_5/dropout/random_uniform/RandomUniform?
-sequential_3/dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2/
-sequential_3/dropout_5/dropout/GreaterEqual/y?
+sequential_3/dropout_5/dropout/GreaterEqualGreaterEqualDsequential_3/dropout_5/dropout/random_uniform/RandomUniform:output:06sequential_3/dropout_5/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2-
+sequential_3/dropout_5/dropout/GreaterEqual?
#sequential_3/dropout_5/dropout/CastCast/sequential_3/dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2%
#sequential_3/dropout_5/dropout/Cast?
$sequential_3/dropout_5/dropout/Mul_1Mul&sequential_3/dropout_5/dropout/Mul:z:0'sequential_3/dropout_5/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2&
$sequential_3/dropout_5/dropout/Mul_1?
*sequential_3/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*sequential_3/dense_4/MatMul/ReadVariableOp?
sequential_3/dense_4/MatMulMatMul(sequential_3/dropout_5/dropout/Mul_1:z:02sequential_3/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_3/dense_4/MatMul?
+sequential_3/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_3/dense_4/BiasAdd/ReadVariableOp?
sequential_3/dense_4/BiasAddBiasAdd%sequential_3/dense_4/MatMul:product:03sequential_3/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_3/dense_4/BiasAdd?
sequential_3/dense_4/ReluRelu%sequential_3/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_3/dense_4/Relu?
*sequential_3/dense_5/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*sequential_3/dense_5/MatMul/ReadVariableOp?
sequential_3/dense_5/MatMulMatMul'sequential_3/dense_4/Relu:activations:02sequential_3/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_3/dense_5/MatMul?
+sequential_3/dense_5/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_3/dense_5/BiasAdd/ReadVariableOp?
sequential_3/dense_5/BiasAddBiasAdd%sequential_3/dense_5/MatMul:product:03sequential_3/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_3/dense_5/BiasAdd?
sequential_3/dense_5/SigmoidSigmoid%sequential_3/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_3/dense_5/Sigmoid?
IdentityIdentity sequential_3/dense_5/Sigmoid:y:0,^sequential_3/dense_4/BiasAdd/ReadVariableOp+^sequential_3/dense_4/MatMul/ReadVariableOp,^sequential_3/dense_5/BiasAdd/ReadVariableOp+^sequential_3/dense_5/MatMul/ReadVariableOp*^sequential_3/embedding_2/embedding_lookupN^text_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:?????????:: :::::2Z
+sequential_3/dense_4/BiasAdd/ReadVariableOp+sequential_3/dense_4/BiasAdd/ReadVariableOp2X
*sequential_3/dense_4/MatMul/ReadVariableOp*sequential_3/dense_4/MatMul/ReadVariableOp2Z
+sequential_3/dense_5/BiasAdd/ReadVariableOp+sequential_3/dense_5/BiasAdd/ReadVariableOp2X
*sequential_3/dense_5/MatMul/ReadVariableOp*sequential_3/dense_5/MatMul/ReadVariableOp2V
)sequential_3/embedding_2/embedding_lookup)sequential_3/embedding_2/embedding_lookup2?
Mtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2Mtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
F
*__inference_dropout_5_layer_call_fn_323754

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_3224762
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_2_layer_call_and_return_conditional_losses_323671

inputs
embedding_lookup_323665
identity??embedding_lookupf
CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:??????????????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_323665Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0**
_class 
loc:@embedding_lookup/323665*4
_output_shapes"
 :??????????????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@embedding_lookup/323665*4
_output_shapes"
 :??????????????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :??????????????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:2$
embedding_lookupembedding_lookup:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
-__inference_sequential_4_layer_call_fn_323031
text_vectorization_1_input
unknown
	unknown_0	
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_3230142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:?????????:: :::::22
StatefulPartitionedCallStatefulPartitionedCall:c _
'
_output_shapes
:?????????
4
_user_specified_nametext_vectorization_1_input:

_output_shapes
: 
?
?
%text_vectorization_1_cond_true_323366E
Atext_vectorization_1_cond_pad_paddings_1_text_vectorization_1_subZ
Vtext_vectorization_1_cond_pad_text_vectorization_1_raggedtotensor_raggedtensortotensor	&
"text_vectorization_1_cond_identity	?
*text_vectorization_1/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2,
*text_vectorization_1/cond/Pad/paddings/1/0?
(text_vectorization_1/cond/Pad/paddings/1Pack3text_vectorization_1/cond/Pad/paddings/1/0:output:0Atext_vectorization_1_cond_pad_paddings_1_text_vectorization_1_sub*
N*
T0*
_output_shapes
:2*
(text_vectorization_1/cond/Pad/paddings/1?
*text_vectorization_1/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*text_vectorization_1/cond/Pad/paddings/0_1?
&text_vectorization_1/cond/Pad/paddingsPack3text_vectorization_1/cond/Pad/paddings/0_1:output:01text_vectorization_1/cond/Pad/paddings/1:output:0*
N*
T0*
_output_shapes

:2(
&text_vectorization_1/cond/Pad/paddings?
text_vectorization_1/cond/PadPadVtext_vectorization_1_cond_pad_text_vectorization_1_raggedtotensor_raggedtensortotensor/text_vectorization_1/cond/Pad/paddings:output:0*
T0	*0
_output_shapes
:??????????????????2
text_vectorization_1/cond/Pad?
"text_vectorization_1/cond/IdentityIdentity&text_vectorization_1/cond/Pad:output:0*
T0	*0
_output_shapes
:??????????????????2$
"text_vectorization_1/cond/Identity"Q
"text_vectorization_1_cond_identity+text_vectorization_1/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?
}
(__inference_dense_4_layer_call_fn_323774

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_3225002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
H__inference_sequential_4_layer_call_and_return_conditional_losses_323014

inputs^
Ztext_vectorization_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handle_
[text_vectorization_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value	
sequential_3_323002
sequential_3_323004
sequential_3_323006
sequential_3_323008
sequential_3_323010
identity??$sequential_3/StatefulPartitionedCall?Mtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2?
 text_vectorization_1/StringLowerStringLowerinputs*'
_output_shapes
:?????????2"
 text_vectorization_1/StringLower?
'text_vectorization_1/StaticRegexReplaceStaticRegexReplace)text_vectorization_1/StringLower:output:0*'
_output_shapes
:?????????*
pattern<br />*
rewrite 2)
'text_vectorization_1/StaticRegexReplace?
)text_vectorization_1/StaticRegexReplace_1StaticRegexReplace0text_vectorization_1/StaticRegexReplace:output:0*'
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite 2+
)text_vectorization_1/StaticRegexReplace_1?
text_vectorization_1/SqueezeSqueeze2text_vectorization_1/StaticRegexReplace_1:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????2
text_vectorization_1/Squeeze?
&text_vectorization_1/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2(
&text_vectorization_1/StringSplit/Const?
.text_vectorization_1/StringSplit/StringSplitV2StringSplitV2%text_vectorization_1/Squeeze:output:0/text_vectorization_1/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:20
.text_vectorization_1/StringSplit/StringSplitV2?
4text_vectorization_1/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        26
4text_vectorization_1/StringSplit/strided_slice/stack?
6text_vectorization_1/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       28
6text_vectorization_1/StringSplit/strided_slice/stack_1?
6text_vectorization_1/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      28
6text_vectorization_1/StringSplit/strided_slice/stack_2?
.text_vectorization_1/StringSplit/strided_sliceStridedSlice8text_vectorization_1/StringSplit/StringSplitV2:indices:0=text_vectorization_1/StringSplit/strided_slice/stack:output:0?text_vectorization_1/StringSplit/strided_slice/stack_1:output:0?text_vectorization_1/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask20
.text_vectorization_1/StringSplit/strided_slice?
6text_vectorization_1/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6text_vectorization_1/StringSplit/strided_slice_1/stack?
8text_vectorization_1/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8text_vectorization_1/StringSplit/strided_slice_1/stack_1?
8text_vectorization_1/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8text_vectorization_1/StringSplit/strided_slice_1/stack_2?
0text_vectorization_1/StringSplit/strided_slice_1StridedSlice6text_vectorization_1/StringSplit/StringSplitV2:shape:0?text_vectorization_1/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_1/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_1/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask22
0text_vectorization_1/StringSplit/strided_slice_1?
Wtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_1/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2Y
Wtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast?
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_1/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2[
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1?
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2c
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape?
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2c
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const?
`text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2b
`text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod?
etext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2g
etext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y?
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2e
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater?
`text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2b
`text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast?
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2e
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1?
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2a
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max?
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2c
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y?
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2a
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add?
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2a
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul?
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2e
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum?
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2e
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum?
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2e
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2?
dtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:?????????2f
dtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount?
^text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2`
^text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis?
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:?????????2[
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum?
btext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2d
btext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0?
^text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2`
^text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis?
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:?????????2[
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat?
Mtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Ztext_vectorization_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handle7text_vectorization_1/StringSplit/StringSplitV2:values:0[text_vectorization_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:?????????2O
Mtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2?
6text_vectorization_1/string_lookup_1/assert_equal/NoOpNoOp*
_output_shapes
 28
6text_vectorization_1/string_lookup_1/assert_equal/NoOp?
-text_vectorization_1/string_lookup_1/IdentityIdentityVtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2/
-text_vectorization_1/string_lookup_1/Identity?
/text_vectorization_1/string_lookup_1/Identity_1Identitybtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:?????????21
/text_vectorization_1/string_lookup_1/Identity_1?
1text_vectorization_1/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 23
1text_vectorization_1/RaggedToTensor/default_value?
)text_vectorization_1/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2+
)text_vectorization_1/RaggedToTensor/Const?
8text_vectorization_1/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_1/RaggedToTensor/Const:output:06text_vectorization_1/string_lookup_1/Identity:output:0:text_vectorization_1/RaggedToTensor/default_value:output:08text_vectorization_1/string_lookup_1/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:??????????????????*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2:
8text_vectorization_1/RaggedToTensor/RaggedTensorToTensor?
text_vectorization_1/ShapeShapeAtext_vectorization_1/RaggedToTensor/RaggedTensorToTensor:result:0*
T0	*
_output_shapes
:2
text_vectorization_1/Shape?
(text_vectorization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(text_vectorization_1/strided_slice/stack?
*text_vectorization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*text_vectorization_1/strided_slice/stack_1?
*text_vectorization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*text_vectorization_1/strided_slice/stack_2?
"text_vectorization_1/strided_sliceStridedSlice#text_vectorization_1/Shape:output:01text_vectorization_1/strided_slice/stack:output:03text_vectorization_1/strided_slice/stack_1:output:03text_vectorization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"text_vectorization_1/strided_slicez
text_vectorization_1/sub/xConst*
_output_shapes
: *
dtype0*
value	B :d2
text_vectorization_1/sub/x?
text_vectorization_1/subSub#text_vectorization_1/sub/x:output:0+text_vectorization_1/strided_slice:output:0*
T0*
_output_shapes
: 2
text_vectorization_1/sub|
text_vectorization_1/Less/yConst*
_output_shapes
: *
dtype0*
value	B :d2
text_vectorization_1/Less/y?
text_vectorization_1/LessLess+text_vectorization_1/strided_slice:output:0$text_vectorization_1/Less/y:output:0*
T0*
_output_shapes
: 2
text_vectorization_1/Less?
text_vectorization_1/condStatelessIftext_vectorization_1/Less:z:0text_vectorization_1/sub:z:0Atext_vectorization_1/RaggedToTensor/RaggedTensorToTensor:result:0*
Tcond0
*
Tin
2	*
Tout
2	*
_lower_using_switch_merge(*0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *9
else_branch*R(
&text_vectorization_1_cond_false_322982*/
output_shapes
:??????????????????*8
then_branch)R'
%text_vectorization_1_cond_true_3229812
text_vectorization_1/cond?
"text_vectorization_1/cond/IdentityIdentity"text_vectorization_1/cond:output:0*
T0	*'
_output_shapes
:?????????d2$
"text_vectorization_1/cond/Identity?
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall+text_vectorization_1/cond/Identity:output:0sequential_3_323002sequential_3_323004sequential_3_323006sequential_3_323008sequential_3_323010*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_3227572&
$sequential_3/StatefulPartitionedCall?
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:0%^sequential_3/StatefulPartitionedCallN^text_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:?????????:: :::::2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall2?
Mtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2Mtext_vectorization_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
3sequential_4_text_vectorization_1_cond_false_3223266
2sequential_4_text_vectorization_1_cond_placeholder~
zsequential_4_text_vectorization_1_cond_strided_slice_sequential_4_text_vectorization_1_raggedtotensor_raggedtensortotensor	3
/sequential_4_text_vectorization_1_cond_identity	?
:sequential_4/text_vectorization_1/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2<
:sequential_4/text_vectorization_1/cond/strided_slice/stack?
<sequential_4/text_vectorization_1/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   2>
<sequential_4/text_vectorization_1/cond/strided_slice/stack_1?
<sequential_4/text_vectorization_1/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2>
<sequential_4/text_vectorization_1/cond/strided_slice/stack_2?
4sequential_4/text_vectorization_1/cond/strided_sliceStridedSlicezsequential_4_text_vectorization_1_cond_strided_slice_sequential_4_text_vectorization_1_raggedtotensor_raggedtensortotensorCsequential_4/text_vectorization_1/cond/strided_slice/stack:output:0Esequential_4/text_vectorization_1/cond/strided_slice/stack_1:output:0Esequential_4/text_vectorization_1/cond/strided_slice/stack_2:output:0*
Index0*
T0	*0
_output_shapes
:??????????????????*

begin_mask*
end_mask26
4sequential_4/text_vectorization_1/cond/strided_slice?
/sequential_4/text_vectorization_1/cond/IdentityIdentity=sequential_4/text_vectorization_1/cond/strided_slice:output:0*
T0	*0
_output_shapes
:??????????????????21
/sequential_4/text_vectorization_1/cond/Identity"k
/sequential_4_text_vectorization_1_cond_identity8sequential_4/text_vectorization_1/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_322544
embedding_2_input
embedding_2_322414
dense_4_322511
dense_4_322513
dense_5_322538
dense_5_322540
identity??dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?!dropout_4/StatefulPartitionedCall?!dropout_5/StatefulPartitionedCall?#embedding_2/StatefulPartitionedCall?
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallembedding_2_inputembedding_2_322414*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_2_layer_call_and_return_conditional_losses_3224052%
#embedding_2/StatefulPartitionedCall?
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall,embedding_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_3224292#
!dropout_4/StatefulPartitionedCall?
*global_average_pooling1d_2/PartitionedCallPartitionedCall*dropout_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *_
fZRX
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_3224522,
*global_average_pooling1d_2/PartitionedCall?
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_2/PartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_3224712#
!dropout_5/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_4_322511dense_4_322513*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_3225002!
dense_4/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_322538dense_5_322540*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_3225272!
dense_5/StatefulPartitionedCall?
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????:::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall:c _
0
_output_shapes
:??????????????????
+
_user_specified_nameembedding_2_input
?
?
%text_vectorization_1_cond_true_323247E
Atext_vectorization_1_cond_pad_paddings_1_text_vectorization_1_subZ
Vtext_vectorization_1_cond_pad_text_vectorization_1_raggedtotensor_raggedtensortotensor	&
"text_vectorization_1_cond_identity	?
*text_vectorization_1/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2,
*text_vectorization_1/cond/Pad/paddings/1/0?
(text_vectorization_1/cond/Pad/paddings/1Pack3text_vectorization_1/cond/Pad/paddings/1/0:output:0Atext_vectorization_1_cond_pad_paddings_1_text_vectorization_1_sub*
N*
T0*
_output_shapes
:2*
(text_vectorization_1/cond/Pad/paddings/1?
*text_vectorization_1/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*text_vectorization_1/cond/Pad/paddings/0_1?
&text_vectorization_1/cond/Pad/paddingsPack3text_vectorization_1/cond/Pad/paddings/0_1:output:01text_vectorization_1/cond/Pad/paddings/1:output:0*
N*
T0*
_output_shapes

:2(
&text_vectorization_1/cond/Pad/paddings?
text_vectorization_1/cond/PadPadVtext_vectorization_1_cond_pad_text_vectorization_1_raggedtotensor_raggedtensortotensor/text_vectorization_1/cond/Pad/paddings:output:0*
T0	*0
_output_shapes
:??????????????????2
text_vectorization_1/cond/Pad?
"text_vectorization_1/cond/IdentityIdentity&text_vectorization_1/cond/Pad:output:0*
T0	*0
_output_shapes
:??????????????????2$
"text_vectorization_1/cond/Identity"Q
"text_vectorization_1_cond_identity+text_vectorization_1/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?
?
-__inference_sequential_4_layer_call_fn_323432

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_3230142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:?????????:: :::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
-__inference_sequential_3_layer_call_fn_323646

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_3225872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
__inference_save_fn_323828
checkpoint_key[
Wstring_lookup_1_index_table_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	??Jstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2?
Jstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Wstring_lookup_1_index_table_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::2L
Jstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2T
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keys2
add/yR
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: 2
addZ
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-values2	
add_1/yX
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: 2
add_1?
IdentityIdentityadd:z:0K^string_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

IdentityO
ConstConst*
_output_shapes
: *
dtype0*
valueB B 2
Const?

Identity_1IdentityConst:output:0K^string_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_1?

Identity_2IdentityQstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2:keys:0K^string_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
:2

Identity_2?

Identity_3Identity	add_1:z:0K^string_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_3S
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 2	
Const_1?

Identity_4IdentityConst_1:output:0K^string_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_4?

Identity_5IdentitySstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2:values:0K^string_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2*
T0	*
_output_shapes
:2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*
_input_shapes
: :2?
Jstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2Jstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?
d
E__inference_dropout_4_layer_call_and_return_conditional_losses_322429

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const?
dropout/MulMulinputsdropout/Const:output:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/Mul_1r
IdentityIdentitydropout/Mul_1:z:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
&text_vectorization_1_cond_false_323367)
%text_vectorization_1_cond_placeholderd
`text_vectorization_1_cond_strided_slice_text_vectorization_1_raggedtotensor_raggedtensortotensor	&
"text_vectorization_1_cond_identity	?
-text_vectorization_1/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2/
-text_vectorization_1/cond/strided_slice/stack?
/text_vectorization_1/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   21
/text_vectorization_1/cond/strided_slice/stack_1?
/text_vectorization_1/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/text_vectorization_1/cond/strided_slice/stack_2?
'text_vectorization_1/cond/strided_sliceStridedSlice`text_vectorization_1_cond_strided_slice_text_vectorization_1_raggedtotensor_raggedtensortotensor6text_vectorization_1/cond/strided_slice/stack:output:08text_vectorization_1/cond/strided_slice/stack_1:output:08text_vectorization_1/cond/strided_slice/stack_2:output:0*
Index0*
T0	*0
_output_shapes
:??????????????????*

begin_mask*
end_mask2)
'text_vectorization_1/cond/strided_slice?
"text_vectorization_1/cond/IdentityIdentity0text_vectorization_1/cond/strided_slice:output:0*
T0	*0
_output_shapes
:??????????????????2$
"text_vectorization_1/cond/Identity"Q
"text_vectorization_1_cond_identity+text_vectorization_1/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?
c
E__inference_dropout_4_layer_call_and_return_conditional_losses_322434

inputs

identity_1g
IdentityIdentityinputs*
T0*4
_output_shapes"
 :??????????????????2

Identityv

Identity_1IdentityIdentity:output:0*
T0*4
_output_shapes"
 :??????????????????2

Identity_1"!

identity_1Identity_1:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
&text_vectorization_1_cond_false_323092)
%text_vectorization_1_cond_placeholderd
`text_vectorization_1_cond_strided_slice_text_vectorization_1_raggedtotensor_raggedtensortotensor	&
"text_vectorization_1_cond_identity	?
-text_vectorization_1/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2/
-text_vectorization_1/cond/strided_slice/stack?
/text_vectorization_1/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   21
/text_vectorization_1/cond/strided_slice/stack_1?
/text_vectorization_1/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/text_vectorization_1/cond/strided_slice/stack_2?
'text_vectorization_1/cond/strided_sliceStridedSlice`text_vectorization_1_cond_strided_slice_text_vectorization_1_raggedtotensor_raggedtensortotensor6text_vectorization_1/cond/strided_slice/stack:output:08text_vectorization_1/cond/strided_slice/stack_1:output:08text_vectorization_1/cond/strided_slice/stack_2:output:0*
Index0*
T0	*0
_output_shapes
:??????????????????*

begin_mask*
end_mask2)
'text_vectorization_1/cond/strided_slice?
"text_vectorization_1/cond/IdentityIdentity0text_vectorization_1/cond/strided_slice:output:0*
T0	*0
_output_shapes
:??????????????????2$
"text_vectorization_1/cond/Identity"Q
"text_vectorization_1_cond_identity+text_vectorization_1/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?
?
%text_vectorization_1_cond_true_322694E
Atext_vectorization_1_cond_pad_paddings_1_text_vectorization_1_subZ
Vtext_vectorization_1_cond_pad_text_vectorization_1_raggedtotensor_raggedtensortotensor	&
"text_vectorization_1_cond_identity	?
*text_vectorization_1/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2,
*text_vectorization_1/cond/Pad/paddings/1/0?
(text_vectorization_1/cond/Pad/paddings/1Pack3text_vectorization_1/cond/Pad/paddings/1/0:output:0Atext_vectorization_1_cond_pad_paddings_1_text_vectorization_1_sub*
N*
T0*
_output_shapes
:2*
(text_vectorization_1/cond/Pad/paddings/1?
*text_vectorization_1/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*text_vectorization_1/cond/Pad/paddings/0_1?
&text_vectorization_1/cond/Pad/paddingsPack3text_vectorization_1/cond/Pad/paddings/0_1:output:01text_vectorization_1/cond/Pad/paddings/1:output:0*
N*
T0*
_output_shapes

:2(
&text_vectorization_1/cond/Pad/paddings?
text_vectorization_1/cond/PadPadVtext_vectorization_1_cond_pad_text_vectorization_1_raggedtotensor_raggedtensortotensor/text_vectorization_1/cond/Pad/paddings:output:0*
T0	*0
_output_shapes
:??????????????????2
text_vectorization_1/cond/Pad?
"text_vectorization_1/cond/IdentityIdentity&text_vectorization_1/cond/Pad:output:0*
T0	*0
_output_shapes
:??????????????????2$
"text_vectorization_1/cond/Identity"Q
"text_vectorization_1_cond_identity+text_vectorization_1/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?
r
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_323711

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?6
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_323603

inputs'
#embedding_2_embedding_lookup_323565*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identity??dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?embedding_2/embedding_lookup~
embedding_2/CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:??????????????????2
embedding_2/Cast?
embedding_2/embedding_lookupResourceGather#embedding_2_embedding_lookup_323565embedding_2/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_2/embedding_lookup/323565*4
_output_shapes"
 :??????????????????*
dtype02
embedding_2/embedding_lookup?
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_2/embedding_lookup/323565*4
_output_shapes"
 :??????????????????2'
%embedding_2/embedding_lookup/Identity?
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :??????????????????2)
'embedding_2/embedding_lookup/Identity_1w
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_4/dropout/Const?
dropout_4/dropout/MulMul0embedding_2/embedding_lookup/Identity_1:output:0 dropout_4/dropout/Const:output:0*
T0*4
_output_shapes"
 :??????????????????2
dropout_4/dropout/Mul?
dropout_4/dropout/ShapeShape0embedding_2/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
dropout_4/dropout/Shape?
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype020
.dropout_4/dropout/random_uniform/RandomUniform?
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 dropout_4/dropout/GreaterEqual/y?
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2 
dropout_4/dropout/GreaterEqual?
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2
dropout_4/dropout/Cast?
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*4
_output_shapes"
 :??????????????????2
dropout_4/dropout/Mul_1?
1global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_2/Mean/reduction_indices?
global_average_pooling1d_2/MeanMeandropout_4/dropout/Mul_1:z:0:global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2!
global_average_pooling1d_2/Meanw
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_5/dropout/Const?
dropout_5/dropout/MulMul(global_average_pooling1d_2/Mean:output:0 dropout_5/dropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout_5/dropout/Mul?
dropout_5/dropout/ShapeShape(global_average_pooling1d_2/Mean:output:0*
T0*
_output_shapes
:2
dropout_5/dropout/Shape?
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype020
.dropout_5/dropout/random_uniform/RandomUniform?
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 dropout_5/dropout/GreaterEqual/y?
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2 
dropout_5/dropout/GreaterEqual?
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout_5/dropout/Cast?
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout_5/dropout/Mul_1?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMuldropout_5/dropout/Mul_1:z:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_4/Relu?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/BiasAddy
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_5/Sigmoid?
IdentityIdentitydense_5/Sigmoid:y:0^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^embedding_2/embedding_lookup*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????:::::2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2<
embedding_2/embedding_lookupembedding_2/embedding_lookup:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
-__inference_sequential_3_layer_call_fn_323546

inputs	
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_3227572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????d:::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
2sequential_4_text_vectorization_1_cond_true_322325_
[sequential_4_text_vectorization_1_cond_pad_paddings_1_sequential_4_text_vectorization_1_subt
psequential_4_text_vectorization_1_cond_pad_sequential_4_text_vectorization_1_raggedtotensor_raggedtensortotensor	3
/sequential_4_text_vectorization_1_cond_identity	?
7sequential_4/text_vectorization_1/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 29
7sequential_4/text_vectorization_1/cond/Pad/paddings/1/0?
5sequential_4/text_vectorization_1/cond/Pad/paddings/1Pack@sequential_4/text_vectorization_1/cond/Pad/paddings/1/0:output:0[sequential_4_text_vectorization_1_cond_pad_paddings_1_sequential_4_text_vectorization_1_sub*
N*
T0*
_output_shapes
:27
5sequential_4/text_vectorization_1/cond/Pad/paddings/1?
7sequential_4/text_vectorization_1/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7sequential_4/text_vectorization_1/cond/Pad/paddings/0_1?
3sequential_4/text_vectorization_1/cond/Pad/paddingsPack@sequential_4/text_vectorization_1/cond/Pad/paddings/0_1:output:0>sequential_4/text_vectorization_1/cond/Pad/paddings/1:output:0*
N*
T0*
_output_shapes

:25
3sequential_4/text_vectorization_1/cond/Pad/paddings?
*sequential_4/text_vectorization_1/cond/PadPadpsequential_4_text_vectorization_1_cond_pad_sequential_4_text_vectorization_1_raggedtotensor_raggedtensortotensor<sequential_4/text_vectorization_1/cond/Pad/paddings:output:0*
T0	*0
_output_shapes
:??????????????????2,
*sequential_4/text_vectorization_1/cond/Pad?
/sequential_4/text_vectorization_1/cond/IdentityIdentity3sequential_4/text_vectorization_1/cond/Pad:output:0*
T0	*0
_output_shapes
:??????????????????21
/sequential_4/text_vectorization_1/cond/Identity"k
/sequential_4_text_vectorization_1_cond_identity8sequential_4/text_vectorization_1/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
a
text_vectorization_1_inputC
,serving_default_text_vectorization_1_input:0?????????@
sequential_30
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?2
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"?0
_tf_keras_sequential?/{"class_name": "Sequential", "name": "sequential_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "text_vectorization_1_input"}}, {"class_name": "TextVectorization", "config": {"name": "text_vectorization_1", "trainable": true, "dtype": "string", "max_tokens": 25000, "standardize": "Custom>custom_standardization", "split": "whitespace", "ngrams": null, "output_mode": "int", "output_sequence_length": 100, "pad_to_max_tokens": true}}, {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "embedding_2_input"}}, {"class_name": "Embedding", "config": {"name": "embedding_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 25001, "output_dim": 16, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "GlobalAveragePooling1D", "config": {"name": "global_average_pooling1d_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "text_vectorization_1_input"}}, {"class_name": "TextVectorization", "config": {"name": "text_vectorization_1", "trainable": true, "dtype": "string", "max_tokens": 25000, "standardize": "Custom>custom_standardization", "split": "whitespace", "ngrams": null, "output_mode": "int", "output_sequence_length": 100, "pad_to_max_tokens": true}}, {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "embedding_2_input"}}, {"class_name": "Embedding", "config": {"name": "embedding_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 25001, "output_dim": 16, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "GlobalAveragePooling1D", "config": {"name": "global_average_pooling1d_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}]}}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "auto", "name": "binary_crossentropy", "from_logits": false, "label_smoothing": 0}}, "metrics": [[{"class_name": "BinaryAccuracy", "config": {"name": "binary_accuracy", "dtype": "float32", "threshold": 0.5}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
?
	state_variables

_index_lookup_layer
	keras_api"?
_tf_keras_layer?{"class_name": "TextVectorization", "name": "text_vectorization_1", "trainable": true, "expects_training_arg": false, "dtype": "string", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "text_vectorization_1", "trainable": true, "dtype": "string", "max_tokens": 25000, "standardize": "Custom>custom_standardization", "split": "whitespace", "ngrams": null, "output_mode": "int", "output_sequence_length": 100, "pad_to_max_tokens": true}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
?)
layer_with_weights-0
layer-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?'
_tf_keras_sequential?'{"class_name": "Sequential", "name": "sequential_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "embedding_2_input"}}, {"class_name": "Embedding", "config": {"name": "embedding_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 25001, "output_dim": 16, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "GlobalAveragePooling1D", "config": {"name": "global_average_pooling1d_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "embedding_2_input"}}, {"class_name": "Embedding", "config": {"name": "embedding_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 25001, "output_dim": 16, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "GlobalAveragePooling1D", "config": {"name": "global_average_pooling1d_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "auto", "name": "binary_crossentropy", "from_logits": true, "label_smoothing": 0}}, "metrics": [[{"class_name": "BinaryAccuracy", "config": {"name": "binary_accuracy", "dtype": "float32", "threshold": 0.05}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0003000000142492354, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
"
	optimizer
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
C
1
2
3
4
5"
trackable_list_wrapper
?
trainable_variables
non_trainable_variables
layer_regularization_losses
regularization_losses
	variables
metrics
layer_metrics

 layers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_dict_wrapper
?
!state_variables

"_table
#	keras_api"?
_tf_keras_layer?{"class_name": "StringLookup", "name": "string_lookup_1", "trainable": true, "expects_training_arg": false, "dtype": "string", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": true, "config": {"name": "string_lookup_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "string", "invert": false, "max_tokens": 25000, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}}
"
_generic_user_object
?

embeddings
$trainable_variables
%regularization_losses
&	variables
'	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 25001, "output_dim": 16, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null]}}
?
(trainable_variables
)regularization_losses
*	variables
+	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?
,trainable_variables
-regularization_losses
.	variables
/	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "GlobalAveragePooling1D", "name": "global_average_pooling1d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_average_pooling1d_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
0trainable_variables
1regularization_losses
2	variables
3	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?

kernel
bias
4trainable_variables
5regularization_losses
6	variables
7	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
?

kernel
bias
8trainable_variables
9regularization_losses
:	variables
;	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
?
<iter

=beta_1

>beta_2
	?decay
@learning_ratemzm{m|m}m~vv?v?v?v?"
	optimizer
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
?
trainable_variables
Anon_trainable_variables
Blayer_regularization_losses
regularization_losses
	variables
Cmetrics
Dlayer_metrics

Elayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(
??2embedding_2/embeddings
 :2dense_4/kernel
:2dense_4/bias
 :2dense_5/kernel
:2dense_5/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
T
?_create_resource
?_initialize
?_destroy_resourceR Z
table??
"
_generic_user_object
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
$trainable_variables
Hnon_trainable_variables
Ilayer_regularization_losses
%regularization_losses
&	variables
Jmetrics
Klayer_metrics

Llayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
(trainable_variables
Mnon_trainable_variables
Nlayer_regularization_losses
)regularization_losses
*	variables
Ometrics
Player_metrics

Qlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
,trainable_variables
Rnon_trainable_variables
Slayer_regularization_losses
-regularization_losses
.	variables
Tmetrics
Ulayer_metrics

Vlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
0trainable_variables
Wnon_trainable_variables
Xlayer_regularization_losses
1regularization_losses
2	variables
Ymetrics
Zlayer_metrics

[layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
4trainable_variables
\non_trainable_variables
]layer_regularization_losses
5regularization_losses
6	variables
^metrics
_layer_metrics

`layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
8trainable_variables
anon_trainable_variables
blayer_regularization_losses
9regularization_losses
:	variables
cmetrics
dlayer_metrics

elayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_dict_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
?
	htotal
	icount
j	variables
k	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	ltotal
	mcount
n
_fn_kwargs
o	variables
p	keras_api"?
_tf_keras_metric?{"class_name": "BinaryAccuracy", "name": "binary_accuracy", "dtype": "float32", "config": {"name": "binary_accuracy", "dtype": "float32", "threshold": 0.5}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?
	qtotal
	rcount
s	variables
t	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	utotal
	vcount
w
_fn_kwargs
x	variables
y	keras_api"?
_tf_keras_metric?{"class_name": "BinaryAccuracy", "name": "binary_accuracy", "dtype": "float32", "config": {"name": "binary_accuracy", "dtype": "float32", "threshold": 0.05}}
:  (2total
:  (2count
.
h0
i1"
trackable_list_wrapper
-
j	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
l0
m1"
trackable_list_wrapper
-
o	variables"
_generic_user_object
:  (2total
:  (2count
.
q0
r1"
trackable_list_wrapper
-
s	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
u0
v1"
trackable_list_wrapper
-
x	variables"
_generic_user_object
/:-
??2Adam/embedding_2/embeddings/m
%:#2Adam/dense_4/kernel/m
:2Adam/dense_4/bias/m
%:#2Adam/dense_5/kernel/m
:2Adam/dense_5/bias/m
/:-
??2Adam/embedding_2/embeddings/v
%:#2Adam/dense_4/kernel/v
:2Adam/dense_4/bias/v
%:#2Adam/dense_5/kernel/v
:2Adam/dense_5/bias/v
?2?
-__inference_sequential_4_layer_call_fn_323451
-__inference_sequential_4_layer_call_fn_323141
-__inference_sequential_4_layer_call_fn_323432
-__inference_sequential_4_layer_call_fn_323031?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_sequential_4_layer_call_and_return_conditional_losses_323413
H__inference_sequential_4_layer_call_and_return_conditional_losses_322920
H__inference_sequential_4_layer_call_and_return_conditional_losses_323308
H__inference_sequential_4_layer_call_and_return_conditional_losses_322829?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
!__inference__wrapped_model_322372?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *9?6
4?1
text_vectorization_1_input?????????
?B?
__inference_save_fn_323828checkpoint_key"?
???
FullArgSpec
args?
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?	
? 
?B?
__inference_restore_fn_323836restored_tensors_0restored_tensors_1"?
???
FullArgSpec
args? 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
	?	
?2?
-__inference_sequential_3_layer_call_fn_322600
-__inference_sequential_3_layer_call_fn_323546
-__inference_sequential_3_layer_call_fn_323561
-__inference_sequential_3_layer_call_fn_323661
-__inference_sequential_3_layer_call_fn_322635
-__inference_sequential_3_layer_call_fn_323646?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_sequential_3_layer_call_and_return_conditional_losses_323631
H__inference_sequential_3_layer_call_and_return_conditional_losses_323531
H__inference_sequential_3_layer_call_and_return_conditional_losses_322544
H__inference_sequential_3_layer_call_and_return_conditional_losses_323502
H__inference_sequential_3_layer_call_and_return_conditional_losses_323603
H__inference_sequential_3_layer_call_and_return_conditional_losses_322564?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
$__inference_signature_wrapper_323162text_vectorization_1_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_embedding_2_layer_call_fn_323678?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_embedding_2_layer_call_and_return_conditional_losses_323671?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dropout_4_layer_call_fn_323705
*__inference_dropout_4_layer_call_fn_323700?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_4_layer_call_and_return_conditional_losses_323690
E__inference_dropout_4_layer_call_and_return_conditional_losses_323695?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
;__inference_global_average_pooling1d_2_layer_call_fn_323716
;__inference_global_average_pooling1d_2_layer_call_fn_323727?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_323722
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_323711?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dropout_5_layer_call_fn_323749
*__inference_dropout_5_layer_call_fn_323754?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_5_layer_call_and_return_conditional_losses_323739
E__inference_dropout_5_layer_call_and_return_conditional_losses_323744?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dense_4_layer_call_fn_323774?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_4_layer_call_and_return_conditional_losses_323765?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_5_layer_call_fn_323794?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_5_layer_call_and_return_conditional_losses_323785?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference__creator_323799?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_323804?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_323809?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
	J
Const7
__inference__creator_323799?

? 
? "? 9
__inference__destroyer_323809?

? 
? "? ;
__inference__initializer_323804?

? 
? "? ?
!__inference__wrapped_model_322372?"?C?@
9?6
4?1
text_vectorization_1_input?????????
? ";?8
6
sequential_3&?#
sequential_3??????????
C__inference_dense_4_layer_call_and_return_conditional_losses_323765\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? {
(__inference_dense_4_layer_call_fn_323774O/?,
%?"
 ?
inputs?????????
? "???????????
C__inference_dense_5_layer_call_and_return_conditional_losses_323785\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? {
(__inference_dense_5_layer_call_fn_323794O/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_dropout_4_layer_call_and_return_conditional_losses_323690v@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
E__inference_dropout_4_layer_call_and_return_conditional_losses_323695v@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
*__inference_dropout_4_layer_call_fn_323700i@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
*__inference_dropout_4_layer_call_fn_323705i@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
E__inference_dropout_5_layer_call_and_return_conditional_losses_323739\3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
E__inference_dropout_5_layer_call_and_return_conditional_losses_323744\3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? }
*__inference_dropout_5_layer_call_fn_323749O3?0
)?&
 ?
inputs?????????
p
? "??????????}
*__inference_dropout_5_layer_call_fn_323754O3?0
)?&
 ?
inputs?????????
p 
? "???????????
G__inference_embedding_2_layer_call_and_return_conditional_losses_323671q8?5
.?+
)?&
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
,__inference_embedding_2_layer_call_fn_323678d8?5
.?+
)?&
inputs??????????????????
? "%?"???????????????????
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_323711i@?=
6?3
-?*
inputs??????????????????

 
? "%?"
?
0?????????
? ?
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_323722{I?F
??<
6?3
inputs'???????????????????????????

 
? ".?+
$?!
0??????????????????
? ?
;__inference_global_average_pooling1d_2_layer_call_fn_323716\@?=
6?3
-?*
inputs??????????????????

 
? "???????????
;__inference_global_average_pooling1d_2_layer_call_fn_323727nI?F
??<
6?3
inputs'???????????????????????????

 
? "!???????????????????z
__inference_restore_fn_323836Y"K?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? ?
__inference_save_fn_323828?"&?#
?
?
checkpoint_key 
? "???
`?]

name?
0/name 
#

slice_spec?
0/slice_spec 

tensor?
0/tensor
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
H__inference_sequential_3_layer_call_and_return_conditional_losses_322544{K?H
A?>
4?1
embedding_2_input??????????????????
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_3_layer_call_and_return_conditional_losses_322564{K?H
A?>
4?1
embedding_2_input??????????????????
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_3_layer_call_and_return_conditional_losses_323502g7?4
-?*
 ?
inputs?????????d	
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_3_layer_call_and_return_conditional_losses_323531g7?4
-?*
 ?
inputs?????????d	
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_3_layer_call_and_return_conditional_losses_323603p@?=
6?3
)?&
inputs??????????????????
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_3_layer_call_and_return_conditional_losses_323631p@?=
6?3
)?&
inputs??????????????????
p 

 
? "%?"
?
0?????????
? ?
-__inference_sequential_3_layer_call_fn_322600nK?H
A?>
4?1
embedding_2_input??????????????????
p

 
? "???????????
-__inference_sequential_3_layer_call_fn_322635nK?H
A?>
4?1
embedding_2_input??????????????????
p 

 
? "???????????
-__inference_sequential_3_layer_call_fn_323546Z7?4
-?*
 ?
inputs?????????d	
p

 
? "???????????
-__inference_sequential_3_layer_call_fn_323561Z7?4
-?*
 ?
inputs?????????d	
p 

 
? "???????????
-__inference_sequential_3_layer_call_fn_323646c@?=
6?3
)?&
inputs??????????????????
p

 
? "???????????
-__inference_sequential_3_layer_call_fn_323661c@?=
6?3
)?&
inputs??????????????????
p 

 
? "???????????
H__inference_sequential_4_layer_call_and_return_conditional_losses_322829~"?K?H
A?>
4?1
text_vectorization_1_input?????????
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_4_layer_call_and_return_conditional_losses_322920~"?K?H
A?>
4?1
text_vectorization_1_input?????????
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_4_layer_call_and_return_conditional_losses_323308j"?7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_4_layer_call_and_return_conditional_losses_323413j"?7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
-__inference_sequential_4_layer_call_fn_323031q"?K?H
A?>
4?1
text_vectorization_1_input?????????
p

 
? "???????????
-__inference_sequential_4_layer_call_fn_323141q"?K?H
A?>
4?1
text_vectorization_1_input?????????
p 

 
? "???????????
-__inference_sequential_4_layer_call_fn_323432]"?7?4
-?*
 ?
inputs?????????
p

 
? "???????????
-__inference_sequential_4_layer_call_fn_323451]"?7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
$__inference_signature_wrapper_323162?"?a?^
? 
W?T
R
text_vectorization_1_input4?1
text_vectorization_1_input?????????";?8
6
sequential_3&?#
sequential_3?????????