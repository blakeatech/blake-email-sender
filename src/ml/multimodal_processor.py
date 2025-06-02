"""
Multimodal Email Processing System
Handles images, attachments, and rich content using vision transformers
"""

import torch
import numpy as np
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    CLIPProcessor, CLIPModel,
    AutoTokenizer, AutoModel
)
from PIL import Image
import cv2
import base64
import io
import os
import mimetypes
from typing import Dict, List, Optional, Union, Tuple
import logging
import requests
from pathlib import Path
import pytesseract
import fitz  # PyMuPDF for PDF processing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalEmailProcessor:
    """Advanced multimodal processing for emails with images and attachments"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load multimodal models
        self._load_models()
        
    def _load_models(self):
        """Load all required multimodal models"""
        try:
            # BLIP for image captioning and VQA
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to(self.device)
            
            # CLIP for image-text similarity
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            
            logger.info("Multimodal models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def process_email_with_attachments(self, email_data: Dict) -> Dict:
        """
        Process email with all its attachments and rich content
        
        Args:
            email_data: Dictionary containing email content and attachments
            
        Returns:
            Comprehensive analysis including multimodal insights
        """
        results = {
            'text_content': email_data.get('body', ''),
            'subject': email_data.get('subject', ''),
            'attachments_analysis': [],
            'images_analysis': [],
            'overall_summary': '',
            'content_type': 'text_only',
            'multimodal_insights': {}
        }
        
        # Process attachments
        attachments = email_data.get('attachments', [])
        if attachments:
            results['content_type'] = 'multimodal'
            
            for attachment in attachments:
                attachment_analysis = self._process_attachment(attachment)
                results['attachments_analysis'].append(attachment_analysis)
                
                # If it's an image, add to images analysis
                if attachment_analysis['type'] == 'image':
                    results['images_analysis'].append(attachment_analysis)
        
        # Process inline images
        inline_images = email_data.get('inline_images', [])
        for image_data in inline_images:
            image_analysis = self._process_image(image_data)
            results['images_analysis'].append(image_analysis)
            results['content_type'] = 'multimodal'
        
        # Generate overall multimodal summary
        if results['content_type'] == 'multimodal':
            results['overall_summary'] = self._generate_multimodal_summary(results)
            results['multimodal_insights'] = self._extract_multimodal_insights(results)
        
        return results
    
    def _process_attachment(self, attachment: Dict) -> Dict:
        """
        Process a single attachment
        
        Args:
            attachment: Dictionary with 'filename', 'content', 'mime_type'
            
        Returns:
            Analysis results for the attachment
        """
        filename = attachment.get('filename', '')
        content = attachment.get('content', b'')
        mime_type = attachment.get('mime_type', '')
        
        analysis = {
            'filename': filename,
            'mime_type': mime_type,
            'size': len(content),
            'type': 'unknown',
            'extracted_text': '',
            'description': '',
            'insights': {}
        }
        
        try:
            # Determine file type and process accordingly
            if mime_type.startswith('image/'):
                analysis['type'] = 'image'
                analysis.update(self._process_image_content(content))
                
            elif mime_type == 'application/pdf':
                analysis['type'] = 'pdf'
                analysis.update(self._process_pdf_content(content))
                
            elif mime_type.startswith('text/'):
                analysis['type'] = 'text'
                analysis['extracted_text'] = content.decode('utf-8', errors='ignore')
                
            elif filename.endswith(('.doc', '.docx')):
                analysis['type'] = 'document'
                analysis.update(self._process_document_content(content, filename))
                
            elif filename.endswith(('.xls', '.xlsx', '.csv')):
                analysis['type'] = 'spreadsheet'
                analysis.update(self._process_spreadsheet_content(content, filename))
                
            else:
                analysis['type'] = 'other'
                analysis['description'] = f"Unsupported file type: {mime_type}"
                
        except Exception as e:
            logger.error(f"Error processing attachment {filename}: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _process_image_content(self, image_content: bytes) -> Dict:
        """Process image content and extract insights"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_content))
            
            return self._process_image(image)
            
        except Exception as e:
            logger.error(f"Error processing image content: {e}")
            return {'error': str(e)}
    
    def _process_image(self, image: Union[Image.Image, str, bytes]) -> Dict:
        """
        Process an image and extract comprehensive information
        
        Args:
            image: PIL Image, file path, or bytes
            
        Returns:
            Dictionary with image analysis results
        """
        try:
            # Convert to PIL Image if necessary
            if isinstance(image, bytes):
                image = Image.open(io.BytesIO(image))
            elif isinstance(image, str):
                image = Image.open(image)
            
            # Ensure RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            analysis = {
                'dimensions': image.size,
                'mode': image.mode,
                'caption': '',
                'objects_detected': [],
                'text_extracted': '',
                'scene_description': '',
                'visual_features': {},
                'content_classification': ''
            }
            
            # Generate image caption using BLIP
            analysis['caption'] = self._generate_image_caption(image)
            
            # Extract text from image using OCR
            analysis['text_extracted'] = self._extract_text_from_image(image)
            
            # Classify image content
            analysis['content_classification'] = self._classify_image_content(image)
            
            # Extract visual features
            analysis['visual_features'] = self._extract_visual_features(image)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {'error': str(e)}
    
    def _generate_image_caption(self, image: Image.Image) -> str:
        """Generate caption for image using BLIP"""
        try:
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=50)
            
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            return caption
            
        except Exception as e:
            logger.error(f"Error generating image caption: {e}")
            return "Unable to generate caption"
    
    def _extract_text_from_image(self, image: Image.Image) -> str:
        """Extract text from image using OCR"""
        try:
            # Convert PIL to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Use Tesseract OCR
            extracted_text = pytesseract.image_to_string(opencv_image)
            
            return extracted_text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return ""
    
    def _classify_image_content(self, image: Image.Image) -> str:
        """Classify image content using CLIP"""
        try:
            # Define possible image categories
            categories = [
                "screenshot", "diagram", "chart", "graph", "document", 
                "photo", "logo", "signature", "table", "flowchart",
                "presentation slide", "invoice", "receipt", "form"
            ]
            
            # Prepare inputs for CLIP
            inputs = self.clip_processor(
                text=[f"a {category}" for category in categories],
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Get the most likely category
            best_category_idx = probs.argmax().item()
            confidence = probs[0][best_category_idx].item()
            
            return f"{categories[best_category_idx]} (confidence: {confidence:.2f})"
            
        except Exception as e:
            logger.error(f"Error classifying image content: {e}")
            return "unknown"
    
    def _extract_visual_features(self, image: Image.Image) -> Dict:
        """Extract visual features from image"""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Basic visual statistics
            features = {
                'mean_brightness': np.mean(img_array),
                'color_variance': np.var(img_array),
                'dominant_colors': self._get_dominant_colors(img_array),
                'has_text': len(self._extract_text_from_image(image)) > 10,
                'aspect_ratio': image.size[0] / image.size[1]
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting visual features: {e}")
            return {}
    
    def _get_dominant_colors(self, img_array: np.ndarray, k: int = 3) -> List[List[int]]:
        """Get dominant colors in the image"""
        try:
            # Reshape image to be a list of pixels
            pixels = img_array.reshape(-1, 3)
            
            # Use k-means clustering to find dominant colors
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get the dominant colors
            colors = kmeans.cluster_centers_.astype(int)
            
            return colors.tolist()
            
        except Exception as e:
            logger.error(f"Error getting dominant colors: {e}")
            return []
    
    def _process_pdf_content(self, pdf_content: bytes) -> Dict:
        """Process PDF content and extract text and images"""
        try:
            # Open PDF from bytes
            pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
            
            analysis = {
                'page_count': len(pdf_document),
                'extracted_text': '',
                'images_found': [],
                'metadata': {}
            }
            
            # Extract text from all pages
            full_text = ""
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                full_text += page.get_text()
                
                # Extract images from page
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(pdf_document, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            image_analysis = self._process_image_content(img_data)
                            analysis['images_found'].append({
                                'page': page_num + 1,
                                'index': img_index,
                                'analysis': image_analysis
                            })
                        
                        pix = None
                    except Exception as e:
                        logger.warning(f"Could not extract image {img_index} from page {page_num}: {e}")
            
            analysis['extracted_text'] = full_text.strip()
            analysis['metadata'] = pdf_document.metadata
            
            pdf_document.close()
            return analysis
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return {'error': str(e)}
    
    def _process_document_content(self, content: bytes, filename: str) -> Dict:
        """Process Word documents (placeholder - would need python-docx)"""
        return {
            'extracted_text': 'Document processing not implemented',
            'note': 'Would require python-docx library for full implementation'
        }
    
    def _process_spreadsheet_content(self, content: bytes, filename: str) -> Dict:
        """Process spreadsheet files (placeholder - would need pandas/openpyxl)"""
        return {
            'extracted_text': 'Spreadsheet processing not implemented',
            'note': 'Would require pandas/openpyxl library for full implementation'
        }
    
    def _generate_multimodal_summary(self, analysis_results: Dict) -> str:
        """Generate a comprehensive summary of multimodal content"""
        try:
            summary_parts = []
            
            # Text content summary
            text_content = analysis_results.get('text_content', '')
            if text_content:
                summary_parts.append(f"Text content discusses: {text_content[:200]}...")
            
            # Images summary
            images = analysis_results.get('images_analysis', [])
            if images:
                image_descriptions = [img.get('caption', 'Unknown image') for img in images]
                summary_parts.append(f"Contains {len(images)} image(s): {', '.join(image_descriptions[:3])}")
            
            # Attachments summary
            attachments = analysis_results.get('attachments_analysis', [])
            if attachments:
                attachment_types = [att.get('type', 'unknown') for att in attachments]
                summary_parts.append(f"Includes {len(attachments)} attachment(s) of types: {', '.join(set(attachment_types))}")
            
            return " | ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating multimodal summary: {e}")
            return "Unable to generate summary"
    
    def _extract_multimodal_insights(self, analysis_results: Dict) -> Dict:
        """Extract insights from multimodal analysis"""
        insights = {
            'has_visual_content': len(analysis_results.get('images_analysis', [])) > 0,
            'has_attachments': len(analysis_results.get('attachments_analysis', [])) > 0,
            'content_complexity': 'simple',
            'visual_text_ratio': 0.0,
            'dominant_content_type': 'text'
        }
        
        try:
            # Calculate visual to text ratio
            text_length = len(analysis_results.get('text_content', ''))
            visual_text_length = sum(
                len(img.get('text_extracted', '')) 
                for img in analysis_results.get('images_analysis', [])
            )
            
            if text_length + visual_text_length > 0:
                insights['visual_text_ratio'] = visual_text_length / (text_length + visual_text_length)
            
            # Determine complexity
            num_attachments = len(analysis_results.get('attachments_analysis', []))
            num_images = len(analysis_results.get('images_analysis', []))
            
            if num_attachments > 3 or num_images > 2:
                insights['content_complexity'] = 'complex'
            elif num_attachments > 1 or num_images > 0:
                insights['content_complexity'] = 'moderate'
            
            # Determine dominant content type
            if insights['visual_text_ratio'] > 0.5:
                insights['dominant_content_type'] = 'visual'
            elif num_attachments > 0:
                insights['dominant_content_type'] = 'mixed'
            
        except Exception as e:
            logger.error(f"Error extracting multimodal insights: {e}")
        
        return insights
    
    def answer_visual_question(self, image: Image.Image, question: str) -> str:
        """Answer questions about images using BLIP VQA"""
        try:
            inputs = self.blip_processor(image, question, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=50)
            
            answer = self.blip_processor.decode(out[0], skip_special_tokens=True)
            return answer
            
        except Exception as e:
            logger.error(f"Error answering visual question: {e}")
            return "Unable to answer question"

# Example usage and testing
if __name__ == "__main__":
    processor = MultimodalEmailProcessor()
    
    # Example email with multimodal content
    sample_email = {
        'subject': 'Project Update with Charts',
        'body': 'Please find attached the latest project charts and screenshots.',
        'attachments': [
            {
                'filename': 'chart.png',
                'content': b'',  # Would contain actual image bytes
                'mime_type': 'image/png'
            }
        ],
        'inline_images': []
    }
    
    # Process the email
    results = processor.process_email_with_attachments(sample_email)
    
    print("Multimodal Email Analysis:")
    print("=" * 50)
    print(f"Content Type: {results['content_type']}")
    print(f"Summary: {results['overall_summary']}")
    print(f"Insights: {results['multimodal_insights']}") 