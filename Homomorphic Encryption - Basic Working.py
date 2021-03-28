# BASIC WORKING OF HOMOMORPHIC ENCRYPTION

# Import Dependencies
import phe as paillier


# Create Public and Private Keys
key_length = 1024
pub_key, private_key = paillier.generate_paillier_keypair(n_length=key_length)

pub_key

private_key


# Encrypt an operation using Public Key
a = 10
print("a: ",a)

encrypted_a = pub_key.encrypt(a)
print("Encrypted a: ",encrypted_a)

print("Encrypted a Public Key: ", encrypted_a.public_key)

# Encrypt another variable
b = 5
print("b: ", b)

encrypted_b = pub_key.encrypt(b)
print("Encrypted b: ", encrypted_b)

print("Encrypted b Public Key: ",encrypted_b.public_key)

# Do an operation on Encrypted Variables
c = a + b
print("c: ", c)

d = a * b
print("d: ",d)

e = a - b

encrypted_e = pub_key.encrypt(e)
print("Encrypted e: ", encrypted_e)

# Decrypt the Encrypted Data
decrypted_e = private_key.decrypt(encrypted_e)

print("Decrypted e: ", decrypted_e)






