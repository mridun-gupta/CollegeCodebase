# Step 1: Initialize the ATM system
# Define a function to handle the ATM operations
def atm_system():
    # Step 2: Prompt user to specify initial balance
    try:
        initial_balance = float(input("Enter initial balance: "))
        # Check if initial balance is non-negative
        if initial_balance < 0:
            print("Error: Initial balance cannot be negative!")
            return
    except ValueError:
        print("Error: Please enter a valid number for the balance!")
        return
    
    # Step 3: Set current balance
    balance = initial_balance
    
    # Step 4: Prompt user to specify withdrawal amount
    try:
        withdraw_amount = float(input("Enter amount to withdraw: "))
        # Check if withdrawal amount is non-negative
        if withdraw_amount < 0:
            print("Error: Withdrawal amount cannot be negative!")
            return
    except ValueError:
        print("Error: Please enter a valid number for the withdrawal amount!")
        return
    
    # Step 5: Check if withdrawal is possible
    if withdraw_amount > balance:
        print("Error: Insufficient balance!")
        print(f"Current balance: {balance:.2f}")
    else:
        # Step 6: Update balance and display result
        balance -= withdraw_amount
        print(f"Withdrawal successful! Amount withdrawn: {withdraw_amount:.2f}")
        print(f"New balance: {balance:.2f}")

# Step 7: Run the ATM system
if __name__ == "__main__":
    print("Welcome to the Basic ATM System")
    atm_system()