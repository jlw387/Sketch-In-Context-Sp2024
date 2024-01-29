import network
import torch

def print_test_result(test_name : str, result : bool):
    print("TEST NAME:", test_name)
    print("RESULT: " + ("SUCCESS" if result else "FAILURE") + "\n")

def run_tests():
    network_default = network.VariationalSketchPretrainer()
    network_two_layer = network.VariationalSketchPretrainer(depth=2)
    network_small_final_grid_size = network.VariationalSketchPretrainer(final_grid_size=4)
    network_small_min_size = network.VariationalSketchPretrainer(depth=3, final_grid_size=4)

    test_results = []
    pass_count = 0

    # TEST 1: Default Minimum Padding Size
    #
    # Checks that a small input is padded to the minimum input size 
    # for the default network
    test_input = torch.tensor([[0]])
    test_output = network.preprocess_input(test_input, 
                                           network_default.min_img_size)
    test_size = test_output.size()

    test_results.append(test_size[-2] == network_default.min_img_size and
                        test_size[-1] == network_default.min_img_size)
    pass_count += int(test_results[-1])

    print_test_result("Default Minimum Padding Size", test_results[-1])
    
    # TEST 2: 2-Layer Minimum Padding Size
    #
    # Checks that a small input is padded to the minimum input size 
    # for a 2-layer network
    test_input = torch.tensor([[0]])
    test_output = network.preprocess_input(test_input, 
                                           network_two_layer.min_img_size)
    test_size = test_output.size()

    test_results.append(test_size[-2] == network_two_layer.min_img_size and
                        test_size[-1] == network_two_layer.min_img_size)
    pass_count += int(test_results[-1])

    print_test_result("2-Layer Minimum Padding Size", test_results[-1])
    
    # TEST 3: Small Final Grid Padding Size
    #
    # Checks that a small input is padded to the minimum input size 
    # for a network with a smaller-than-defult final grid size
    test_input = torch.tensor([[0]])
    test_output = network.preprocess_input(test_input, 
                                           network_small_final_grid_size.min_img_size)
    test_size = test_output.size()

    test_results.append(test_size[-2] == network_small_final_grid_size.min_img_size and
                        test_size[-1] == network_small_final_grid_size.min_img_size)
    pass_count += int(test_results[-1])

    print_test_result("Small Final Grid Minimum Padding Size", test_results[-1]) 

    # TEST 6: Odd-Size Input Padding Location
    #
    # Checks that a 29x23 input image is padded to the next power of two using a network
    # a small minimum input size (or at least checks a number of sample points of the 
    # expected result)
    test_input = torch.full(size=(1,1,23,29),fill_value=1)
    test_output = network.preprocess_input(test_input,
                                           network_small_min_size.min_img_size)
    test_size = test_output.size()

    top_left_check = test_output[0,0,0,0] == 0
    top_right_check = test_output[0,0,0,-1] == 0
    bottom_left_check = test_output[0,0,-1,0] == 0
    bottom_right_check = test_output[0,0,-1,-1] == 0
    
    middle_check = test_output[0,0,int(test_size[-2] / 2),int(test_size[-1]/2)] == 1

    tlo_check = test_output[0,0,3,0] == 0
    tli_check = test_output[0,0,4,1] == 1

    tro_check = test_output[0,0,3,-2] == 0
    tri_check = test_output[0,0,4,-3] == 1

    blo_check = test_output[0,0,-5,0] == 0
    bli_check = test_output[0,0,-6,1] == 1

    bro_check = test_output[0,0,-5,-2] == 0
    bri_check = test_output[0,0,-6,-3] == 1

    test_results.append(top_left_check and top_right_check and
                        bottom_left_check and bottom_right_check and
                        middle_check and
                        tlo_check and tli_check and tro_check and tri_check and
                        blo_check and bli_check and bro_check and bri_check)
    pass_count += int(test_results[-1])

    print_test_result("Odd-Size Input Padding Location", test_results[-1])

    # Final printout
    print("TESTS PASSED:", pass_count, "of", len(test_results))


if __name__ == "__main__":
    run_tests()

     