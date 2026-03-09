def generate_data_with_gan(constraints: list, gan_model):
    """Generate data using a GAN model based on SHACL constraints"""
    
    generated_data = []
    for constraint in constraints:
        
        generated_data.append(gan_model.generate(constraint))
    return generated_data
