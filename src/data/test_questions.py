"""
Generate test questions with ground truth for evaluation.
"""
import json
from pathlib import Path
from typing import List, Dict, Any


def generate_test_dataset() -> List[Dict[str, Any]]:
    """
    Generate test questions with ground truth answers and expected retrieved docs.

    Distribution:
    - 60% (15) single-hop answerable questions
    - 20% (5) multi-hop reasoning questions
    - 20% (5) unanswerable questions (not in corpus)
    """

    test_cases = [
        # ===== SINGLE-HOP ANSWERABLE QUESTIONS (15) =====
        {
            "id": "test_001",
            "question": "What is your return policy?",
            "expected_answer": "Most items can be returned within 30 days of delivery for a full refund. Items must be unused, in original packaging, and in the same condition as received.",
            "expected_doc_ids": ["faq_001"],
            "category": "returns_refunds",
            "type": "single_hop",
            "answerable": True
        },
        {
            "id": "test_002",
            "question": "How do I reset my password?",
            "expected_answer": "Click 'Forgot Password' on the login page, enter your email address, and we'll send you a password reset link. The link expires after 24 hours.",
            "expected_doc_ids": ["faq_022"],
            "category": "account_management",
            "type": "single_hop",
            "answerable": True
        },
        {
            "id": "test_003",
            "question": "How long does standard shipping take?",
            "expected_answer": "Standard shipping takes 5-7 business days for delivery within the continental US. Orders are processed within 1-2 business days.",
            "expected_doc_ids": ["faq_011"],
            "category": "shipping_tracking",
            "type": "single_hop",
            "answerable": True
        },
        {
            "id": "test_004",
            "question": "What payment methods do you accept?",
            "expected_answer": "We accept Visa, Mastercard, American Express, Discover, PayPal, Apple Pay, Google Pay, and Amazon Pay. Gift cards and store credit can also be used.",
            "expected_doc_ids": ["faq_041"],
            "category": "payment_billing",
            "type": "single_hop",
            "answerable": True
        },
        {
            "id": "test_005",
            "question": "How do I track my order?",
            "expected_answer": "Log into your account and view 'Order History'. Click on the order number to see detailed tracking information. You can also track using the tracking number sent to your email through the carrier's website.",
            "expected_doc_ids": ["faq_015"],
            "category": "shipping_tracking",
            "type": "single_hop",
            "answerable": True
        },
        {
            "id": "test_006",
            "question": "What items cannot be returned?",
            "expected_answer": "Personalized or custom-made products, perishable goods, intimate apparel, hazardous materials, digital downloads, and gift cards cannot be returned.",
            "expected_doc_ids": ["faq_004"],
            "category": "returns_refunds",
            "type": "single_hop",
            "answerable": True
        },
        {
            "id": "test_007",
            "question": "How much does Premium membership cost?",
            "expected_answer": "Premium membership costs $99 per year and includes free shipping on all orders, early access to sales, exclusive discounts, and priority customer service.",
            "expected_doc_ids": ["faq_026"],
            "category": "account_management",
            "type": "single_hop",
            "answerable": True
        },
        {
            "id": "test_008",
            "question": "Do you offer free shipping?",
            "expected_answer": "Yes, orders over $50 qualify for free standard shipping within the US (excluding Alaska, Hawaii, and US territories). Premium members receive free shipping on all orders.",
            "expected_doc_ids": ["faq_013"],
            "category": "shipping_tracking",
            "type": "single_hop",
            "answerable": True
        },
        {
            "id": "test_009",
            "question": "How do I initiate a return?",
            "expected_answer": "Log into your account, go to 'Order History', find the order you want to return, click 'Return Items', select the items and reason for return, then print the prepaid return shipping label.",
            "expected_doc_ids": ["faq_002"],
            "category": "returns_refunds",
            "type": "single_hop",
            "answerable": True
        },
        {
            "id": "test_010",
            "question": "What happens if my payment is declined?",
            "expected_answer": "Verify your billing information matches your bank records, check that you have sufficient funds, and contact your bank as they may have flagged the transaction. You can try a different payment method.",
            "expected_doc_ids": ["faq_049"],
            "category": "payment_billing",
            "type": "single_hop",
            "answerable": True
        },
        {
            "id": "test_011",
            "question": "Can I ship to a PO Box?",
            "expected_answer": "Yes, we can ship to PO boxes using USPS only. Expedited shipping is not available for PO box addresses, and large or oversized items cannot be delivered to PO boxes.",
            "expected_doc_ids": ["faq_018"],
            "category": "shipping_tracking",
            "type": "single_hop",
            "answerable": True
        },
        {
            "id": "test_012",
            "question": "What is covered by the product warranty?",
            "expected_answer": "Most products come with a 1-year manufacturer warranty against defects. Electronics include a 2-year warranty. Warranty does not cover normal wear and tear or accidental damage.",
            "expected_doc_ids": ["faq_034"],
            "category": "product_info",
            "type": "single_hop",
            "answerable": True
        },
        {
            "id": "test_013",
            "question": "When will I be charged for my order?",
            "expected_answer": "Your payment method is authorized at checkout but only charged when your order ships. For orders with multiple shipments, you'll be charged separately for each package as it ships.",
            "expected_doc_ids": ["faq_043"],
            "category": "payment_billing",
            "type": "single_hop",
            "answerable": True
        },
        {
            "id": "test_014",
            "question": "How do I delete my account?",
            "expected_answer": "Go to 'Account Settings' and click 'Delete Account'. This action is permanent and cannot be undone. All order history, saved addresses, and payment methods will be removed.",
            "expected_doc_ids": ["faq_025"],
            "category": "account_management",
            "type": "single_hop",
            "answerable": True
        },
        {
            "id": "test_015",
            "question": "Do you ship internationally?",
            "expected_answer": "Yes, we ship to over 100 countries worldwide. International shipping costs vary by destination and package weight. Delivery times range from 7-21 business days. Customers are responsible for customs duties, taxes, or fees.",
            "expected_doc_ids": ["faq_014"],
            "category": "shipping_tracking",
            "type": "single_hop",
            "answerable": True
        },

        # ===== MULTI-HOP REASONING QUESTIONS (5) =====
        {
            "id": "test_016",
            "question": "If I'm a Premium member, how much will it cost me to return an item I don't like?",
            "expected_answer": "Premium members receive free return shipping on all items, so it will cost you nothing to return an item you don't like (as long as it's within 30 days and meets return conditions).",
            "expected_doc_ids": ["faq_003", "faq_026"],  # Return shipping + Premium benefits
            "category": "multi_hop",
            "type": "multi_hop",
            "answerable": True,
            "reasoning": "Requires combining info about return shipping costs and Premium membership benefits"
        },
        {
            "id": "test_017",
            "question": "How long will it take to get my money back if I return something today?",
            "expected_answer": "It takes 2-3 business days to inspect your return after we receive it, then 5-7 business days to process the refund. Your bank may take an additional 3-5 days to post the credit, so total time is approximately 10-15 business days.",
            "expected_doc_ids": ["faq_005"],  # Refund processing time
            "category": "returns_refunds",
            "type": "multi_hop",
            "answerable": True,
            "reasoning": "Requires adding up multiple time periods mentioned in the refund processing doc"
        },
        {
            "id": "test_018",
            "question": "I need an item urgently tomorrow and I'm not a Premium member. What are my total costs for a $40 item?",
            "expected_answer": "You would need overnight shipping ($24.99). The item is $40, and since it's under $50, standard shipping wouldn't be free anyway. Your total would be $40 + $24.99 = $64.99 plus applicable sales tax.",
            "expected_doc_ids": ["faq_012", "faq_013"],  # Expedited shipping + Free shipping threshold
            "category": "multi_hop",
            "type": "multi_hop",
            "answerable": True,
            "reasoning": "Requires understanding overnight shipping cost and that free shipping threshold doesn't apply"
        },
        {
            "id": "test_019",
            "question": "Can I change my shipping address after placing an order, and if so, how long do I have?",
            "expected_answer": "You can change your shipping address within 2 hours of placing your order by contacting customer service. After that, once the order enters processing, address changes are not possible.",
            "expected_doc_ids": ["faq_017"],  # Change shipping address
            "category": "shipping_tracking",
            "type": "multi_hop",
            "answerable": True,
            "reasoning": "Requires understanding the time constraint and the process"
        },
        {
            "id": "test_020",
            "question": "If I receive a damaged item, do I have to pay for return shipping and how quickly should I report it?",
            "expected_answer": "No, you don't have to pay for return shipping. We will provide a prepaid return label. You should contact customer service within 48 hours of delivery with photos of the damage.",
            "expected_doc_ids": ["faq_008", "faq_003"],  # Damaged items + Return shipping costs
            "category": "multi_hop",
            "type": "multi_hop",
            "answerable": True,
            "reasoning": "Combines damaged item policy with return shipping cost information"
        },

        # ===== UNANSWERABLE QUESTIONS (5) =====
        {
            "id": "test_021",
            "question": "What are your store hours?",
            "expected_answer": "I don't have information about store hours in our FAQ database. This information is not available.",
            "expected_doc_ids": [],
            "category": "unanswerable",
            "type": "unanswerable",
            "answerable": False,
            "reasoning": "No physical store information in e-commerce FAQ"
        },
        {
            "id": "test_022",
            "question": "Do you offer installation services for appliances?",
            "expected_answer": "I don't have information about installation services in our FAQ database. This information is not available.",
            "expected_doc_ids": [],
            "category": "unanswerable",
            "type": "unanswerable",
            "answerable": False,
            "reasoning": "Installation services not mentioned in FAQ corpus"
        },
        {
            "id": "test_023",
            "question": "What is your company's sustainability policy?",
            "expected_answer": "I don't have information about sustainability policies in our FAQ database. This information is not available.",
            "expected_doc_ids": [],
            "category": "unanswerable",
            "type": "unanswerable",
            "answerable": False,
            "reasoning": "Sustainability not covered in customer support FAQs"
        },
        {
            "id": "test_024",
            "question": "Can I rent products instead of buying them?",
            "expected_answer": "I don't have information about product rental options in our FAQ database. This information is not available.",
            "expected_doc_ids": [],
            "category": "unanswerable",
            "type": "unanswerable",
            "answerable": False,
            "reasoning": "Rental services not mentioned in FAQ"
        },
        {
            "id": "test_025",
            "question": "What COVID-19 safety measures do you have in your warehouses?",
            "expected_answer": "I don't have information about COVID-19 safety measures in our FAQ database. This information is not available.",
            "expected_doc_ids": [],
            "category": "unanswerable",
            "type": "unanswerable",
            "answerable": False,
            "reasoning": "COVID-19 policies not in FAQ corpus"
        }
    ]

    return test_cases


def save_test_dataset(output_path: str = "data/eval_dataset.json"):
    """Generate and save the test dataset to a JSON file."""
    test_data = generate_test_dataset()

    # Create data directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(test_data)} test questions")
    print(f"Saved to {output_path}")

    # Print distribution
    from collections import Counter
    types = Counter(q['type'] for q in test_data)
    print("\nQuestion type distribution:")
    for qtype, count in types.items():
        print(f"  {qtype}: {count} questions ({count/len(test_data)*100:.0f}%)")

    answerable_count = sum(1 for q in test_data if q['answerable'])
    print(f"\nAnswerable: {answerable_count} ({answerable_count/len(test_data)*100:.0f}%)")
    print(f"Unanswerable: {len(test_data) - answerable_count} ({(len(test_data) - answerable_count)/len(test_data)*100:.0f}%)")

    return test_data


if __name__ == "__main__":
    save_test_dataset()
