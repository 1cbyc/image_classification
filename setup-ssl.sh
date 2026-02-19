#!/bin/bash
# SSL Setup Script for ReluRay
# Run this as root or with sudo

set -e

echo "🔒 Setting up SSL for ReluRay..."
echo "========================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "⚠️  Please run as root or with sudo"
    exit 1
fi

# Check if certbot is installed
if ! command -v certbot &> /dev/null; then
    echo "📦 Installing certbot..."
    apt-get update
    apt-get install -y certbot python3-certbot-nginx
fi

# Check if nginx is installed
if ! command -v nginx &> /dev/null; then
    echo "📦 Installing nginx..."
    apt-get install -y nginx
fi

echo ""
echo "🌐 Obtaining SSL certificate..."
echo "========================================"
echo "This will obtain a free SSL certificate from Let's Encrypt"
echo "Make sure:"
echo "1. Domain reluray.com points to this server"
echo "2. Port 80 is open to the internet"
echo "3. No other web server is using port 80"
echo ""

read -p "Continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ SSL setup cancelled"
    exit 1
fi

# Obtain SSL certificate
echo "📝 Obtaining certificate for reluray.com and www.reluray.com..."
certbot certonly --nginx -d reluray.com -d www.reluray.com --non-interactive --agree-tos --email ei@nsisong.com

if [ $? -eq 0 ]; then
    echo "✅ SSL certificate obtained successfully!"
    
    # Update nginx config
    echo ""
    echo "🔄 Updating nginx configuration..."
    
    # Backup existing config
    if [ -f "/etc/nginx/sites-available/reluray" ]; then
        cp /etc/nginx/sites-available/reluray /etc/nginx/sites-available/reluray.backup.$(date +%Y%m%d)
    fi
    
    # Copy new config
    cp nginx-https.conf /etc/nginx/sites-available/reluray
    
    # Enable site
    ln -sf /etc/nginx/sites-available/reluray /etc/nginx/sites-enabled/
    
    # Test nginx config
    echo "🧪 Testing nginx configuration..."
    nginx -t
    
    if [ $? -eq 0 ]; then
        echo "✅ nginx configuration is valid"
        
        # Reload nginx
        echo "🔄 Reloading nginx..."
        systemctl reload nginx
        
        echo ""
        echo "========================================"
        echo "🎉 SSL Setup Complete!"
        echo "========================================"
        echo ""
        echo "✅ HTTPS is now enabled at: https://reluray.com"
        echo "✅ HTTP redirects to HTTPS"
        echo "✅ Security headers configured"
        echo "✅ API documentation available at: https://reluray.com/api/docs"
        echo ""
        echo "📋 Next steps:"
        echo "1. Visit https://reluray.com to verify SSL works"
        echo "2. Check security at: https://www.ssllabs.com/ssltest/"
        echo "3. Set up auto-renewal: certbot renew --dry-run"
        echo ""
        echo "🔧 Auto-renewal setup (optional):"
        echo "Add to crontab (crontab -e):"
        echo "0 12 * * * /usr/bin/certbot renew --quiet"
        echo ""
        
    else
        echo "❌ nginx configuration test failed"
        echo "Restoring backup..."
        if [ -f "/etc/nginx/sites-available/reluray.backup" ]; then
            cp /etc/nginx/sites-available/reluray.backup /etc/nginx/sites-available/reluray
            nginx -t && systemctl reload nginx
        fi
        exit 1
    fi
    
else
    echo "❌ Failed to obtain SSL certificate"
    echo "Possible issues:"
    echo "1. Domain not pointing to this server"
    echo "2. Port 80 blocked by firewall"
    echo "3. Another web server using port 80"
    echo ""
    echo "You can manually obtain certificates:"
    echo "certbot certonly --standalone -d reluray.com -d www.reluray.com"
    exit 1
fi