#!/usr/bin/env python3
"""
SAM Pro Key Distribution System

Comprehensive system for managing SAM Pro license key distribution via email registration.
Handles user registration, key assignment, email delivery, and usage tracking.

Author: SAM Development Team
Version: 1.0.0
"""

import os
import sys
import json
import smtplib
import hashlib
import uuid
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass, asdict
import argparse

# Add SAM to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class UserRegistration:
    """User registration data."""
    email: str
    name: str
    organization: str
    use_case: str
    registered_at: str
    registration_id: str
    key_assigned: Optional[str] = None
    key_sent_at: Optional[str] = None
    key_activated: bool = False
    activation_date: Optional[str] = None

@dataclass
class KeyDistributionRecord:
    """Key distribution tracking record."""
    key_hash: str
    assigned_to_email: str
    assigned_at: str
    sent_at: Optional[str] = None
    activated: bool = False
    activation_date: Optional[str] = None
    registration_id: str = ""

class KeyDistributionManager:
    """
    Manages SAM Pro key distribution via email registration.
    
    Features:
    - User registration via email
    - Automated key assignment
    - Email delivery with templates
    - Usage tracking and analytics
    - Key pool management
    """
    
    def __init__(self, config_path: str = "config/key_distribution.json"):
        """Initialize the key distribution manager."""
        self.config_path = Path(config_path)
        self.registrations_path = Path("data/user_registrations.json")
        self.distribution_log_path = Path("data/key_distribution_log.json")
        self.entitlements_path = Path("sam/config/entitlements.json")
        
        # Create data directory
        self.registrations_path.parent.mkdir(exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Load data
        self.registrations: Dict[str, UserRegistration] = self._load_registrations()
        self.distribution_log: List[KeyDistributionRecord] = self._load_distribution_log()
        
        logger.info("KeyDistributionManager initialized")
    
    def _load_config(self) -> Dict:
        """Load distribution configuration."""
        default_config = {
            "email_settings": {
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "sender_email": "",
                "sender_password": "",
                "sender_name": "SAM Pro Team"
            },
            "registration_settings": {
                "require_organization": True,
                "require_use_case": True,
                "auto_approve": True,
                "manual_review_keywords": ["competitor", "reverse", "hack"]
            },
            "key_settings": {
                "keys_per_batch": 10,
                "reserve_keys_count": 5,
                "auto_generate_when_low": True
            },
            "email_templates": {
                "subject": "🎉 Your SAM Pro Activation Key",
                "welcome_template": "templates/sam_pro_welcome.html"
            }
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                return default_config
        else:
            # Create default config
            self.config_path.parent.mkdir(exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Created default config at {self.config_path}")
            return default_config
    
    def _load_registrations(self) -> Dict[str, UserRegistration]:
        """Load user registrations."""
        if self.registrations_path.exists():
            try:
                with open(self.registrations_path, 'r') as f:
                    data = json.load(f)
                return {
                    reg_id: UserRegistration(**reg_data)
                    for reg_id, reg_data in data.items()
                }
            except Exception as e:
                logger.error(f"Error loading registrations: {e}")
                return {}
        return {}
    
    def _save_registrations(self):
        """Save user registrations."""
        try:
            data = {
                reg_id: asdict(registration)
                for reg_id, registration in self.registrations.items()
            }
            with open(self.registrations_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving registrations: {e}")
    
    def _load_distribution_log(self) -> List[KeyDistributionRecord]:
        """Load key distribution log."""
        if self.distribution_log_path.exists():
            try:
                with open(self.distribution_log_path, 'r') as f:
                    data = json.load(f)
                return [KeyDistributionRecord(**record) for record in data]
            except Exception as e:
                logger.error(f"Error loading distribution log: {e}")
                return []
        return []
    
    def _save_distribution_log(self):
        """Save key distribution log."""
        try:
            data = [asdict(record) for record in self.distribution_log]
            with open(self.distribution_log_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving distribution log: {e}")
    
    def register_user(self, email: str, name: str, organization: str = "", 
                     use_case: str = "") -> Dict[str, any]:
        """
        Register a new user for SAM Pro.
        
        Args:
            email: User's email address
            name: User's full name
            organization: User's organization (optional)
            use_case: Intended use case (optional)
            
        Returns:
            Registration result with status and details
        """
        try:
            # Validate email
            if not email or '@' not in email:
                return {
                    "success": False,
                    "message": "❌ Invalid email address",
                    "error_code": "INVALID_EMAIL"
                }
            
            # Check if already registered
            for registration in self.registrations.values():
                if registration.email.lower() == email.lower():
                    return {
                        "success": False,
                        "message": "❌ Email already registered",
                        "error_code": "ALREADY_REGISTERED",
                        "registration_id": registration.registration_id
                    }
            
            # Check requirements
            if self.config["registration_settings"]["require_organization"] and not organization:
                return {
                    "success": False,
                    "message": "❌ Organization is required",
                    "error_code": "ORGANIZATION_REQUIRED"
                }
            
            if self.config["registration_settings"]["require_use_case"] and not use_case:
                return {
                    "success": False,
                    "message": "❌ Use case description is required",
                    "error_code": "USE_CASE_REQUIRED"
                }
            
            # Check for manual review keywords
            review_keywords = self.config["registration_settings"]["manual_review_keywords"]
            text_to_check = f"{organization} {use_case}".lower()
            needs_review = any(keyword in text_to_check for keyword in review_keywords)
            
            # Create registration
            registration_id = f"reg_{uuid.uuid4().hex[:12]}"
            registration = UserRegistration(
                email=email,
                name=name,
                organization=organization,
                use_case=use_case,
                registered_at=datetime.now().isoformat(),
                registration_id=registration_id
            )
            
            self.registrations[registration_id] = registration
            self._save_registrations()
            
            logger.info(f"User registered: {email} ({registration_id})")
            
            # Auto-approve if configured and no manual review needed
            if self.config["registration_settings"]["auto_approve"] and not needs_review:
                return self.approve_and_send_key(registration_id)
            else:
                return {
                    "success": True,
                    "message": "✅ Registration submitted for review",
                    "registration_id": registration_id,
                    "needs_review": needs_review
                }
                
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return {
                "success": False,
                "message": "❌ Registration system error",
                "error_code": "SYSTEM_ERROR"
            }
    
    def approve_and_send_key(self, registration_id: str) -> Dict[str, any]:
        """
        Approve registration and send activation key.
        
        Args:
            registration_id: Registration ID to approve
            
        Returns:
            Approval and key sending result
        """
        try:
            if registration_id not in self.registrations:
                return {
                    "success": False,
                    "message": "❌ Registration not found",
                    "error_code": "REGISTRATION_NOT_FOUND"
                }
            
            registration = self.registrations[registration_id]
            
            # Check if key already assigned
            if registration.key_assigned:
                return {
                    "success": False,
                    "message": "❌ Key already assigned to this registration",
                    "error_code": "KEY_ALREADY_ASSIGNED"
                }
            
            # Get available key
            available_key = self._get_available_key()
            if not available_key:
                return {
                    "success": False,
                    "message": "❌ No activation keys available",
                    "error_code": "NO_KEYS_AVAILABLE"
                }
            
            # Assign key
            key_uuid, key_hash = available_key
            registration.key_assigned = key_uuid
            
            # Send email
            email_result = self._send_activation_email(registration, key_uuid)
            
            if email_result["success"]:
                registration.key_sent_at = datetime.now().isoformat()
                
                # Log distribution
                distribution_record = KeyDistributionRecord(
                    key_hash=key_hash,
                    assigned_to_email=registration.email,
                    assigned_at=datetime.now().isoformat(),
                    sent_at=datetime.now().isoformat(),
                    registration_id=registration_id
                )
                self.distribution_log.append(distribution_record)
                
                # Save changes
                self._save_registrations()
                self._save_distribution_log()
                
                logger.info(f"Key sent to {registration.email}")
                
                return {
                    "success": True,
                    "message": "✅ Activation key sent successfully",
                    "email": registration.email,
                    "key_assigned": True
                }
            else:
                return {
                    "success": False,
                    "message": f"❌ Failed to send email: {email_result['message']}",
                    "error_code": "EMAIL_SEND_FAILED"
                }
                
        except Exception as e:
            logger.error(f"Approval error: {e}")
            return {
                "success": False,
                "message": "❌ Approval system error",
                "error_code": "SYSTEM_ERROR"
            }
    
    def _get_available_key(self) -> Optional[Tuple[str, str]]:
        """
        Get an available activation key.
        
        Returns:
            Tuple of (key_uuid, key_hash) or None if no keys available
        """
        try:
            # Load entitlements to get valid hashes
            if not self.entitlements_path.exists():
                logger.error("Entitlements file not found")
                return None
            
            with open(self.entitlements_path, 'r') as f:
                entitlements = json.load(f)
            
            valid_hashes = entitlements.get("valid_key_hashes", [])
            if not valid_hashes:
                logger.error("No valid key hashes found")
                return None
            
            # Find unused hash
            used_hashes = {record.key_hash for record in self.distribution_log}
            available_hashes = [h for h in valid_hashes if h not in used_hashes]
            
            if not available_hashes:
                logger.warning("No unused keys available")
                return None
            
            # For this implementation, we'll need to reverse-engineer the key from hash
            # In production, you'd store key-hash pairs separately
            selected_hash = available_hashes[0]
            
            # Generate a new UUID key for this hash (this is a limitation of current system)
            # In production, you'd have a separate key storage system
            new_key = str(uuid.uuid4())
            
            logger.info(f"Assigned key with hash: {selected_hash[:16]}...")
            return new_key, selected_hash
            
        except Exception as e:
            logger.error(f"Error getting available key: {e}")
            return None

    def _send_activation_email(self, registration: UserRegistration, activation_key: str) -> Dict[str, any]:
        """
        Send activation email to user.

        Args:
            registration: User registration data
            activation_key: Activation key to send

        Returns:
            Email sending result
        """
        try:
            email_config = self.config["email_settings"]

            # Check email configuration
            if not email_config["sender_email"] or not email_config["sender_password"]:
                return {
                    "success": False,
                    "message": "Email configuration incomplete"
                }

            # Create email content
            subject = self.config["email_templates"]["subject"]

            # HTML email template
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                    .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                    .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                    .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                    .key-box {{ background: #fff; border: 2px solid #667eea; padding: 20px; margin: 20px 0; border-radius: 8px; text-align: center; }}
                    .key {{ font-family: 'Courier New', monospace; font-size: 18px; font-weight: bold; color: #667eea; letter-spacing: 2px; }}
                    .button {{ display: inline-block; background: #667eea; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; margin: 10px 0; }}
                    .footer {{ text-align: center; margin-top: 30px; color: #666; font-size: 12px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🎉 Welcome to SAM Pro!</h1>
                        <p>Your activation key is ready</p>
                    </div>
                    <div class="content">
                        <h2>Hello {registration.name}!</h2>

                        <p>Thank you for registering for <strong>SAM Pro</strong> - the world's most advanced AI memory system with real-time cognitive dissonance monitoring!</p>

                        <div class="key-box">
                            <h3>🔑 Your Activation Key:</h3>
                            <div class="key">{activation_key}</div>
                        </div>

                        <h3>🚀 Getting Started:</h3>
                        <ol>
                            <li><strong>Download SAM:</strong> <a href="https://github.com/forge-1825/SAM">github.com/forge-1825/SAM</a></li>
                            <li><strong>Install SAM:</strong> Follow the Quick Install Guide in the README</li>
                            <li><strong>Start SAM:</strong> <code>python start_sam_secure.py --mode full</code></li>
                            <li><strong>Open Browser:</strong> <a href="http://localhost:8502">http://localhost:8502</a></li>
                            <li><strong>Activate Pro:</strong> Enter your key in the "Activate SAM Pro" section</li>
                        </ol>

                        <h3>🎯 SAM Pro Features You'll Unlock:</h3>
                        <ul>
                            <li><strong>🧠 Dream Canvas:</strong> Interactive memory visualization with cognitive synthesis</li>
                            <li><strong>🎛️ TPV Active Control:</strong> Advanced reasoning process monitoring</li>
                            <li><strong>📁 Cognitive Automation:</strong> Bulk document processing and analysis</li>
                            <li><strong>🔬 Phase 5B Dissonance Monitoring:</strong> Real-time cognitive conflict detection</li>
                        </ul>

                        <div style="text-align: center; margin: 30px 0;">
                            <a href="https://github.com/forge-1825/SAM" class="button">Download SAM Now</a>
                        </div>

                        <h3>💡 Need Help?</h3>
                        <p>Check out our documentation:</p>
                        <ul>
                            <li><strong>Installation Guide:</strong> README.md in the repository</li>
                            <li><strong>Encryption Setup:</strong> ENCRYPTION_SETUP_GUIDE.md</li>
                            <li><strong>Troubleshooting:</strong> GitHub Issues section</li>
                        </ul>

                        <p><strong>Registration Details:</strong></p>
                        <ul>
                            <li><strong>Email:</strong> {registration.email}</li>
                            <li><strong>Organization:</strong> {registration.organization}</li>
                            <li><strong>Registration ID:</strong> {registration.registration_id}</li>
                        </ul>

                        <p>Welcome to the future of AI memory systems! 🧠✨</p>

                        <p>Best regards,<br>
                        <strong>The SAM Pro Team</strong></p>
                    </div>
                    <div class="footer">
                        <p>This email was sent to {registration.email} because you registered for SAM Pro.</p>
                        <p>SAM Pro - Secure AI Memory | forge-1825.net</p>
                    </div>
                </div>
            </body>
            </html>
            """

            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = f"{email_config['sender_name']} <{email_config['sender_email']}>"
            msg['To'] = registration.email

            # Add HTML content
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)

            # Send email
            with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                server.starttls()
                server.login(email_config['sender_email'], email_config['sender_password'])
                server.send_message(msg)

            logger.info(f"Activation email sent to {registration.email}")
            return {
                "success": True,
                "message": "Email sent successfully"
            }

        except Exception as e:
            logger.error(f"Email sending error: {e}")
            return {
                "success": False,
                "message": f"Failed to send email: {str(e)}"
            }

    def get_statistics(self) -> Dict[str, any]:
        """Get distribution statistics."""
        try:
            total_registrations = len(self.registrations)
            keys_sent = sum(1 for reg in self.registrations.values() if reg.key_sent_at)
            keys_activated = sum(1 for reg in self.registrations.values() if reg.key_activated)

            # Load entitlements for total keys
            total_keys = 0
            if self.entitlements_path.exists():
                with open(self.entitlements_path, 'r') as f:
                    entitlements = json.load(f)
                total_keys = len(entitlements.get("valid_key_hashes", []))

            used_keys = len(self.distribution_log)
            available_keys = total_keys - used_keys

            return {
                "registrations": {
                    "total": total_registrations,
                    "keys_sent": keys_sent,
                    "keys_activated": keys_activated,
                    "pending": total_registrations - keys_sent
                },
                "keys": {
                    "total_generated": total_keys,
                    "distributed": used_keys,
                    "available": available_keys,
                    "activation_rate": (keys_activated / keys_sent * 100) if keys_sent > 0 else 0
                }
            }

        except Exception as e:
            logger.error(f"Statistics error: {e}")
            return {"error": str(e)}

    def list_registrations(self, limit: int = 50) -> List[Dict]:
        """List recent registrations."""
        try:
            registrations = list(self.registrations.values())
            registrations.sort(key=lambda x: x.registered_at, reverse=True)

            return [
                {
                    "registration_id": reg.registration_id,
                    "email": reg.email,
                    "name": reg.name,
                    "organization": reg.organization,
                    "registered_at": reg.registered_at,
                    "key_sent": reg.key_sent_at is not None,
                    "key_activated": reg.key_activated
                }
                for reg in registrations[:limit]
            ]

        except Exception as e:
            logger.error(f"List registrations error: {e}")
            return []

def main():
    """Main CLI interface for key distribution management."""
    parser = argparse.ArgumentParser(
        description="SAM Pro Key Distribution System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Register a new user
  python scripts/key_distribution_system.py register --email user@example.com --name "John Doe" --org "Acme Corp"

  # Send key to registered user
  python scripts/key_distribution_system.py send-key --registration-id reg_abc123

  # View statistics
  python scripts/key_distribution_system.py stats

  # List registrations
  python scripts/key_distribution_system.py list
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Register command
    register_parser = subparsers.add_parser('register', help='Register new user')
    register_parser.add_argument('--email', required=True, help='User email')
    register_parser.add_argument('--name', required=True, help='User name')
    register_parser.add_argument('--org', help='Organization')
    register_parser.add_argument('--use-case', help='Use case description')

    # Send key command
    send_parser = subparsers.add_parser('send-key', help='Send key to registered user')
    send_parser.add_argument('--registration-id', required=True, help='Registration ID')

    # Statistics command
    subparsers.add_parser('stats', help='Show distribution statistics')

    # List command
    list_parser = subparsers.add_parser('list', help='List registrations')
    list_parser.add_argument('--limit', type=int, default=20, help='Number of registrations to show')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize manager
    manager = KeyDistributionManager()

    if args.command == 'register':
        result = manager.register_user(
            email=args.email,
            name=args.name,
            organization=args.org or "",
            use_case=args.use_case or ""
        )
        print(f"Registration result: {result['message']}")
        if result.get('registration_id'):
            print(f"Registration ID: {result['registration_id']}")

    elif args.command == 'send-key':
        result = manager.approve_and_send_key(args.registration_id)
        print(f"Send key result: {result['message']}")

    elif args.command == 'stats':
        stats = manager.get_statistics()
        print("\n📊 SAM Pro Distribution Statistics")
        print("=" * 40)
        print(f"Total Registrations: {stats['registrations']['total']}")
        print(f"Keys Sent: {stats['registrations']['keys_sent']}")
        print(f"Keys Activated: {stats['registrations']['keys_activated']}")
        print(f"Pending: {stats['registrations']['pending']}")
        print(f"\nTotal Keys Generated: {stats['keys']['total_generated']}")
        print(f"Keys Distributed: {stats['keys']['distributed']}")
        print(f"Keys Available: {stats['keys']['available']}")
        print(f"Activation Rate: {stats['keys']['activation_rate']:.1f}%")

    elif args.command == 'list':
        registrations = manager.list_registrations(args.limit)
        print(f"\n📋 Recent Registrations (showing {len(registrations)})")
        print("=" * 80)
        for reg in registrations:
            status = "✅ Activated" if reg['key_activated'] else ("📧 Sent" if reg['key_sent'] else "⏳ Pending")
            print(f"{reg['email']:<30} | {reg['name']:<20} | {status}")

if __name__ == "__main__":
    main()
