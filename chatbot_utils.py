import os
import boto3
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)


def ask_receipt_chatbot(question, region_name='us-east-1', table_name='Receipts'):
    """
    Given a user question, fetch all receipts from DynamoDB, format them as context, and query Perplexity LLM.
    Returns the LLM's answer or an error message.
    """
    logger.info(f"[chatbot] Received chatbot question: {question}")
    # Initialize OpenAI client for Perplexity
    api_key = os.environ.get('PERPLEXITY_API_KEY') or os.environ.get('OPENAI_API_KEY')
    if api_key:
        logger.info("[chatbot] Using Perplexity/OpenAI API key from environment.")
    else:
        logger.warning("[chatbot] No Perplexity/OpenAI API key found in environment!")
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.perplexity.ai"
    )
    try:
        logger.info(f"[chatbot] Connecting to DynamoDB in region: {region_name}")
        dynamodb = boto3.resource('dynamodb', region_name=region_name)
        logger.info(f"[chatbot] Getting table: {table_name}")
        table = dynamodb.Table(table_name)
        logger.info(f"[chatbot] Scanning DynamoDB table: {table_name}")
        response = table.scan()
        items = response.get('Items', [])
        logger.info(f"[chatbot] Fetched {len(items)} items from DynamoDB. Response: {response}")
        # Format items for context
        context = "\n".join([
            f"Receipt {item.get('receipt_no', '')}:\n"
            f"  Name: {item.get('name', '')}\n"
            f"  Date: {item.get('date', '')}\n"
            f"  Product: {item.get('product', '')}\n"
            f"  Amount Paid: {item.get('amount_paid', '')}\n"
            for item in items
        ])
        logger.info(f"[chatbot] Context for LLM prompt created. Length: {len(context)} characters. Context: {context}")
        prompt = f"Based on these receipts:\n{context}\n\nQuestion: {question}\nPlease provide a 2-3 line answer."
        logger.info(f"[chatbot] Prompt for LLM: {prompt}")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an artificial intelligence assistant and you need to "
                    "engage in a helpful, detailed, polite conversation with a user. "
                    "Give a 2-3 line answer."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        logger.info("[chatbot] Sending request to Perplexity LLM...")
        response = client.chat.completions.create(
            model="sonar",
            messages=messages
        )
        logger.info(f"[chatbot] Received response from Perplexity LLM: {response}")
        answer = response.choices[0].message.content
        logger.info(f"[chatbot] LLM answer: {answer}")
        return answer
    except Exception as e:
        logger.error(f"[chatbot] Error in ask_receipt_chatbot: {str(e)}", exc_info=True)
        return f"Error from LLM or DynamoDB: {str(e)}" 