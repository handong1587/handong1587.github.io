var undef;

// Some shims for IE
if (!Object.keys) {
    Object.keys = function(obj) {
        var arr = [];
        for (var key in obj) {
            arr.push(key);
        }

        return arr;
    };
}


var opts = {
	step: 100
}

test("encode_id_with_array ", function() {
	var r1 = encode_id_with_array(opts,[1 ]);
	ok( r1 == 1, "[1] = 1 = 1 = 1");
	
	var r1_2 = encode_id_with_array(opts,[2 ]);
	ok( r1_2 == 2, "[2] = 2 = 2 = 2");
	
	var r2 = encode_id_with_array(opts,[1 ,1]);
	ok( r2 == 101, "[1 ,1 ] = 1.1 =  1*100 + 1 = 101");
	 
	var r2_2 = encode_id_with_array(opts,[1 ,2]);
    ok( r2_2 == 102, "[1 ,2] = 1.2 =  1*100 + 2 = 102");
	
	var r2_3 = encode_id_with_array(opts,[2 ,3]);
    ok( r2_3 == 203, "[2 ,3] = 2.3 =  2*100 + 3 = 203");
	
	var r3 = encode_id_with_array(opts,[1 ,1 ,1]);      
	ok( r3 == 10101, "[1 ,1 ,1] = 1.1.1 = 1*100*100 + 1*100 + 1 = 10101");
	
	var r3_2_3 = encode_id_with_array(opts,[1 ,2 ,3]);
    ok( r3_2_3 == 10203, "[1 ,2, 3] = 1.2.3 =  1*100*100 + 2*100 + 3= 10203");
});


test("get_parent_id_with_array ", function() {
	var r1 = get_parent_id_with_array(opts,[1]);
	ok( r1 == 0, "[1] pid = 0");
	
	var r1_2 = get_parent_id_with_array(opts,[2 ]);
	ok( r1_2 == 0, "[2] pid = 0");
	
	var r2 = get_parent_id_with_array(opts,[1 ,1]);
	ok( r2 == 1, "[1 ,1] pid = 1");
	 
	var r2_2 = get_parent_id_with_array(opts,[1 ,2]);
    ok( r2_2 == 1, "[1 ,2] pid = 1");
	
	var r3 = get_parent_id_with_array(opts,[1 ,1 ,1]);      
	ok( r3 == 101, "[1 ,1 ,1] pid = 101");
	
	var r3_2_3 = get_parent_id_with_array(opts,[1 ,2 ,3]);
    ok( r3_2_3 == 102, "[1 ,2, 3] pid = 102");
	
	var r3_2_3_1 = get_parent_id_with_array(opts,[2 ,3 ,1]);
    ok( r3_2_3_1 == 203, "[2, 3, 1] pid = 203");
});

test("factor util method", function() {
	var r3 = factor(opts,3,1);
	var r2 = factor(opts,2,1);
     
    ok( r3 == 100*100, "m(opts,3) = 100*100");
	ok( r2 == 100, "m(opts,2) = 100");
});
