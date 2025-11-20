"""
    Using the BORN effective charges.

    Natural (eV) units are used throughout unless otherwise specified
"""

"""
    Dictionary containing tree level c coefficients, which determines which operators 
    contribute to a scattering process. Numbering follows the convention in 
    the paper. 

    To include different oeprators simply change the value of c_dict.

    Note: This should only contain constant values. If you want to include 
    q/m_chi dependence add it to c_dict_form below
"""

c_dict = {
	1: {
            "e": 1,
            "p": -1,
            "n": 0
	},
	3: {
            "e": 0,
            "p": 0,
            "n": 0
	},
	4: {
            "e": 0,
            "p": 0,
            "n": 0
	},
	5: {
            "e": 0,
            "p": 0,
            "n": 0
	},
	6: {
            "e": 0,
            "p": 0,
            "n": 0
	},
	7: {
            "e": 0,
            "p": 0,
            "n": 0
	},
	8: {
            "e": 0,
            "p": 0,
            "n": 0
	},
	9: {
            "e": 0,
            "p": 0,
            "n": 0
	},
	10: {
            "e": 0,
            "p": 0,
            "n": 0
	},
	11: {
            "e": 0,
            "p": 0,
            "n": 0
	},
}


def c_dict_form(op_id, particle_id, q_vec, mass, spin):
    """
        q/m_chi dependence of the c coefficients. 

        Input:
            op_id : integer, operator id number
            particle_id : string, {"e", "p", "n"} for electron, proton, neutron resp.

            q_vec : (real, real, real), momentum vector in XYZ coordinates
            mass : dark matter mass
            spin : dark matter spin

        Output:
            real, the q/m_chi dependence of the c coefficients that isn't stored above in 
            c_dict


        Note: To add different operators simply add more functions inside of here, and replace
            one_func in the output dict
    """
    def one_func(q_vec, mass, spin):
        return 1.0

    return {
            1: {
                "e": one_func,
                "p": one_func,
                "n": one_func
            },
            3: {
                "e": one_func,
                "p": one_func,
                "n": one_func
            },
            4: {
                "e": one_func,
                "p": one_func,
                "n": one_func
            },
            5: {
                "e": one_func,
                "p": one_func,
                "n": one_func
            },
            6: {
                "e": one_func,
                "p": one_func,
                "n": one_func
            },
            7: {
                "e": one_func,
                "p": one_func,
                "n": one_func
            },
            8: {
                "e": one_func,
                "p": one_func,
                "n": one_func
            },
            9: {
                "e": one_func,
                "p": one_func,
                "n": one_func
            },
            10: {
                "e": one_func,
                "p": one_func,
                "n": one_func
            },
            11: {
                "e": one_func,
                "p": one_func,
                "n": one_func
            },
        }[op_id][particle_id](q_vec, mass, spin)
